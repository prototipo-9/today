
import React, { useState, useRef, useCallback, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI, LiveSession, LiveServerMessage, Modality, FunctionDeclaration, Type, Blob } from '@google/genai';

// =================================================================
// TYPES (from types.ts)
// =================================================================
enum Status {
  IDLE = 'IDLE',
  LISTENING = 'LISTENING',
}

type AppStatus = Status.IDLE | Status.LISTENING;

type PhaseInfo = {
  name: string;
  age: string;
};

type TextTranscriptEntry = {
  author: 'user' | 'model';
  text: string;
  type: 'text';
}

type ImageTranscriptEntry = {
  author: 'model';
  imageUrl: string;
  prompt: string;
  type: 'image';
}

type PronunciationGuidanceEntry = {
  author: 'model';
  word: string;
  approximation: string;
  explanation: string;
  type: 'pronunciation';
}

type VideoTranscriptEntry = {
  author: 'model';
  videoUrl: string;
  prompt: string;
  type: 'video';
}

type TranscriptEntry = TextTranscriptEntry | ImageTranscriptEntry | PronunciationGuidanceEntry | VideoTranscriptEntry;


// =================================================================
// AUDIO SERVICES (from services/audio.ts)
// =================================================================

/**
 * Encodes a Uint8Array into a Base64 string.
 */
function encode(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Decodes a Base64 string into a Uint8Array.
 */
function decode(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

/**
 * Decodes raw PCM audio data into an AudioBuffer for playback.
 */
async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

/**
 * Creates a Gemini API-compatible Blob from raw audio data.
 */
function createBlob(data: Float32Array): Blob {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}


// =================================================================
// APP COMPONENT (from App.tsx)
// =================================================================
const MicIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3zM18.999 13a1 1 0 0 0-1 1a6 6 0 0 1-12 0a1 1 0 1 0-2 0a8 8 0 0 0 7 7.93V24h2v-2.07A8 8 0 0 0 19.999 14a1 1 0 0 0-1-1z" />
  </svg>
);

const StopIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
    <path d="M6 6h12v12H6z" />
  </svg>
);

const showImageFunctionDeclaration: FunctionDeclaration = {
  name: 'show_image',
  description: 'Gere uma imagem simples, clara e didática adequada à fase atual do aluno para ajudar a transmitir significado.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      prompt: {
        type: Type.STRING,
        description: 'Um prompt em inglês, simples e descritivo para a imagem. Ex: "a red ball", "a cat is sleeping".',
      },
    },
    required: ['prompt'],
  },
};

const explainPronunciationFunctionDeclaration: FunctionDeclaration = {
  name: 'explain_pronunciation',
  description: 'Explica a pronúncia de uma palavra em inglês usando aproximações em português e descrevendo a articulação da boca.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      word: { type: Type.STRING, description: 'A palavra em inglês.' },
      approximation_pt_br: { type: Type.STRING, description: 'A aproximação sonora em português (BR). Ex: "uóter".' },
      explanation_pt_br: { type: Type.STRING, description: 'A explicação de como mover a boca, lábios e língua, seguindo o modelo definido.' },
    },
    required: ['word', 'approximation_pt_br', 'explanation_pt_br'],
  },
};

const showArticulationVideoFunctionDeclaration: FunctionDeclaration = {
  name: 'show_articulation_video',
  description: 'Gera um vídeo em close da boca pronunciando uma palavra ou frase lentamente para demonstrar a articulação.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      prompt: {
        type: Type.STRING,
        description: 'Prompt para o vídeo, seguindo o formato: "Crie um vídeo educacional em close mostrando apenas a boca e o rosto neutro de uma pessoa, fundo claro, pronunciando lentamente a palavra [PALAVRA EM INGLÊS]. Mostre claramente a posição dos lábios e da língua. Sem símbolos técnicos."',
      },
    },
    required: ['prompt'],
  },
};

const updatePhaseFunctionDeclaration: FunctionDeclaration = {
  name: 'update_phase',
  description: 'Atualiza a fase de desenvolvimento e idade linguística do aluno na interface.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      phase_name: { type: Type.STRING, description: 'O nome da fase atual. Ex: "FASE 1 — BEBÊ".' },
      linguistic_age: { type: Type.STRING, description: 'A idade linguística aproximada. Ex: "0–2 anos".' },
    },
    required: ['phase_name', 'linguistic_age'],
  },
};


const MAX_RETRIES = 3;
const INITIAL_RETRY_DELAY_MS = 1000;

const App: React.FC = () => {
  const [status, setStatus] = useState<AppStatus>(Status.IDLE);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);
  const [phase, setPhase] = useState<PhaseInfo>({ name: 'FASE 1 — BEBÊ', age: '0–2 anos' });
  const [hasSelectedKey, setHasSelectedKey] = useState(false);

  const sessionPromiseRef = useRef<Promise<LiveSession> | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  
  const currentInputTranscriptionRef = useRef<string>('');
  const currentOutputTranscriptionRef = useRef<string>('');
  const nextStartTimeRef = useRef<number>(0);
  const audioSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const retryTimeoutRef = useRef<number | null>(null);
  const isClosingRef = useRef(false);

  useEffect(() => {
    const checkKey = async () => {
        if ((window as any).aistudio && await (window as any).aistudio.hasSelectedApiKey()) {
            setHasSelectedKey(true);
        }
    };
    checkKey();
  }, []);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcript, isGeneratingImage, isGeneratingVideo]);

  const cleanUpAudio = useCallback(() => {
    scriptProcessorRef.current?.disconnect();
    scriptProcessorRef.current = null;
    mediaStreamSourceRef.current?.disconnect();
    mediaStreamSourceRef.current = null;
    inputAudioContextRef.current?.close().catch(console.error);
    inputAudioContextRef.current = null;
    outputAudioContextRef.current?.close().catch(console.error);
    outputAudioContextRef.current = null;
    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    mediaStreamRef.current = null;
    
    audioSourcesRef.current.forEach(source => source.stop());
    audioSourcesRef.current.clear();
    nextStartTimeRef.current = 0;
  }, []);
  
  const stopConversation = useCallback(async () => {
    isClosingRef.current = true;
    setStatus(Status.IDLE);

    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }

    if (sessionPromiseRef.current) {
        try {
            const session = await sessionPromiseRef.current;
            session.close();
        } catch (e) {
            console.error("Error closing session:", e);
        } finally {
            sessionPromiseRef.current = null;
        }
    }
    cleanUpAudio();
    setTimeout(() => { isClosingRef.current = false; }, 200);
  }, [cleanUpAudio]);

  const startConversation = useCallback(async () => {
    isClosingRef.current = false;

    const attemptConnection = async (retryCount = 0) => {
        if (retryCount === 0) {
            setError(null);
            setStatus(Status.LISTENING);
        }
    
        try {
          const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          mediaStreamRef.current = stream;

          inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
          outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
          
          const sessionPromise = ai.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-12-2025',
            config: {
              responseModalities: [Modality.AUDIO],
              speechConfig: {
                voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } },
              },
              inputAudioTranscription: {},
              outputAudioTranscription: {},
              tools: [{ functionDeclarations: [showImageFunctionDeclaration, explainPronunciationFunctionDeclaration, showArticulationVideoFunctionDeclaration, updatePhaseFunctionDeclaration] }],
              systemInstruction: `Você é uma professora-mãe brasileira, extremamente paciente, acolhedora e especializada em aquisição natural de linguagem (Comprehensible Input).
====================
CONTEXTO DO ALUNO
====================
- O aluno entende português (BR) fluentemente.
- O inglês é absolutamente novo.
- O aprendizado simula o crescimento humano real.
- Tudo acontece como conversação em tempo real (voz + texto).
- O aprendizado usa imagens e vídeos gerados por IA.
- O aluno nunca deve sentir pressão ou cobrança.

====================
REGRAS ABSOLUTAS
====================
1. Nunca use IPA.
2. Nunca use símbolos fonéticos.
3. Use APENAS aproximações de som em português (BR).
4. A aproximação em português permanece em TODAS as fases, inclusive adulta e acadêmica.
5. Explique sempre a pronúncia descrevendo:
   - movimento da boca
   - posição dos lábios
   - posição da língua
   - força ou suavidade do som
6. Fale SEMPRE em português (BR), inclusive quando o aluno acerta.
7. O inglês aparece apenas como input natural (fala, texto, imagens).
8. Nunca diga “está errado”. Corrija apenas reformulando naturalmente.

====================
EXIBIÇÃO DA FASE (EM TEMPO REAL)
====================
Antes de cada interação, use a função 'update_phase' para mostrar:
[Fase atual simulada: ___ | Idade linguística aproximada: ___]

A progressão acontece gradualmente durante a conversa.

====================
MODELO FIXO DE EXPLICAÇÃO DE PRONÚNCIA
====================
Use a função 'explain_pronunciation' com este modelo:
“Esse som começa parecido com ‘___’ em português, mas termina mais fraco.
Os lábios fazem ___, a língua fica ___.
Não precisa forçar.”

====================
FASES + EXERCÍCIOS EXECUTADOS AO VIVO
====================

--------------------------------------------------
FASE 1 — BEBÊ
Idade linguística: 0–2 anos
Objetivo: associação som ↔ imagem
--------------------------------------------------
- Apresente UMA palavra por vez.
- Gere UMA imagem simples e clara.
- Pronuncie lentamente.
- Explique o som em português.
- Exercícios:
  • apontar imagem correta
  • reagir emocionalmente
  • ouvir repetição natural

--------------------------------------------------
FASE 2 — CRIANÇA PEQUENA
Idade: 3–5 anos
Objetivo: frases mínimas
--------------------------------------------------
- Frases curtíssimas.
- Histórias simples com imagens.
- Exercícios:
  • escolher imagens
  • responder com 1 palavra
  • completar frases ouvindo

--------------------------------------------------
FASE 3 — CRIANÇA MAIOR
Idade: 6–9 anos
Objetivo: frases completas simples
--------------------------------------------------
- Diálogos curtos.
- Situações do cotidiano.
- Exercícios:
  • descrever imagens
  • simular situações
  • reformular ouvindo

--------------------------------------------------
FASE 4 — ADOLESCENTE
Idade: 10–15 anos
Objetivo: fluidez e naturalidade
--------------------------------------------------
- Conversas reais.
- Emoções e opiniões.
- Exercícios:
  • simulações sociais
  • narrativas pessoais
  • diferença entre informal e mais cuidadoso

--------------------------------------------------
FASE 5 — ADULTO
Objetivo: inglês funcional completo
--------------------------------------------------
- Conversas longas.
- Contextos profissionais.
- Exercícios:
  • entrevistas simuladas
  • reuniões
  • storytelling

A pronúncia continua explicada em português, com mais refinamento.

--------------------------------------------------
FASE 6 — ACADÊMICA
Objetivo: precisão e registro formal
--------------------------------------------------
- Vocabulário técnico.
- Textos e apresentações.
- Exercícios:
  • resumir textos
  • explicar conceitos
  • ajustar tom acadêmico

Mesmo aqui, a pronúncia continua sendo explicada em português (BR).

====================
IMAGENS (OBRIGATÓRIO)
====================
Sempre que introduzir:
- palavra
- ação
- emoção
- objeto
Use a função 'show_image' com um prompt adequado.

====================
VÍDEOS DE ARTICULAÇÃO (PROMPT INTERNO)
====================
Quando necessário, use a função 'show_articulation_video' com o prompt:
“Crie um vídeo educacional em close mostrando apenas a boca e o rosto neutro de uma pessoa, fundo claro, pronunciando lentamente a palavra ou frase em inglês. Mostre claramente a posição dos lábios e da língua. Sem símbolos técnicos.”

====================
TOM DA PROFESSORA (MÃE)
====================
- Sempre calma.
- Sempre acolhedora.
- Sempre em português (BR).
- Nunca apressa.
- Nunca corrige diretamente.

Exemplo:
“Muito bem. Mesmo assim, vamos continuar devagar. Ouvir bastante é o mais importante.”

====================
INÍCIO
====================
Inicie agora na FASE DE BEBÊ, mostrando a fase atual, gerando a primeira imagem, explicando o som em português e aplicando o primeiro exercício.`,
            },
            callbacks: {
              onopen: () => {
                if (!inputAudioContextRef.current || !mediaStreamRef.current) return;
                mediaStreamSourceRef.current = inputAudioContextRef.current.createMediaStreamSource(mediaStreamRef.current);
                scriptProcessorRef.current = inputAudioContextRef.current.createScriptProcessor(4096, 1, 1);
                
                scriptProcessorRef.current.onaudioprocess = (audioProcessingEvent) => {
                    const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                    const pcmBlob = createBlob(inputData);
                    sessionPromiseRef.current?.then((session) => {
                        session.sendRealtimeInput({ media: pcmBlob });
                    });
                };
                
                mediaStreamSourceRef.current.connect(scriptProcessorRef.current);
                scriptProcessorRef.current.connect(inputAudioContextRef.current.destination);
              },
              onmessage: async (message: LiveServerMessage) => {
                if (message.serverContent?.inputTranscription) {
                  currentInputTranscriptionRef.current += message.serverContent.inputTranscription.text;
                }
                if (message.serverContent?.outputTranscription) {
                  currentOutputTranscriptionRef.current += message.serverContent.outputTranscription.text;
                }

                if (message.toolCall) {
                  for (const fc of message.toolCall.functionCalls) {
                    if (fc.name === 'update_phase' && fc.args.phase_name && fc.args.linguistic_age) {
                        setPhase({ name: fc.args.phase_name as string, age: fc.args.linguistic_age as string });
                        sessionPromiseRef.current?.then(s => s.sendToolResponse({ functionResponses: { id: fc.id, name: fc.name, response: { result: "Phase updated" } } }));
                    } else if (fc.name === 'explain_pronunciation') {
                        setTranscript(prev => [...prev, {
                            type: 'pronunciation', author: 'model',
                            word: fc.args.word as string,
                            approximation: fc.args.approximation_pt_br as string,
                            explanation: fc.args.explanation_pt_br as string
                        }]);
                        sessionPromiseRef.current?.then(s => s.sendToolResponse({ functionResponses: { id: fc.id, name: fc.name, response: { result: "Pronunciation explained" } } }));
                    } else if (fc.name === 'show_image' && fc.args.prompt) {
                      const prompt = fc.args.prompt as string;
                      setIsGeneratingImage(true);
                      try {
                        const imageResponse = await ai.models.generateContent({ model: 'gemini-2.5-flash-image', contents: { parts: [{ text: prompt }] } });
                        let imageUrl: string | null = null;
                        for (const part of imageResponse.candidates[0].content.parts) {
                          if (part.inlineData) {
                            imageUrl = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
                            break;
                          }
                        }
                        if (imageUrl) {
                          setTranscript(prev => [...prev, { type: 'image', author: 'model', imageUrl, prompt }]);
                        }
                        sessionPromiseRef.current?.then(s => s.sendToolResponse({ functionResponses: { id: fc.id, name: fc.name, response: { result: `Displayed image for: ${prompt}` } } }));
                      } catch (imgErr) {
                        console.error("Image generation failed:", imgErr);
                        sessionPromiseRef.current?.then(s => s.sendToolResponse({ functionResponses: { id: fc.id, name: fc.name, response: { result: `Failed to show image for: ${prompt}` } } }));
                      } finally {
                        setIsGeneratingImage(false);
                      }
                    } else if (fc.name === 'show_articulation_video' && fc.args.prompt) {
                        const prompt = fc.args.prompt as string;
                        setIsGeneratingVideo(true);
                        try {
                            const videoAi = new GoogleGenAI({ apiKey: process.env.API_KEY });
                            let operation = await videoAi.models.generateVideos({ model: 'veo-3.1-fast-generate-preview', prompt, config: { numberOfVideos: 1, resolution: '720p', aspectRatio: '16:9' } });
                            while (!operation.done) {
                                await new Promise(resolve => setTimeout(resolve, 5000));
                                operation = await videoAi.operations.getVideosOperation({ operation });
                            }
                            const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;
                            if (downloadLink) {
                                const videoResponse = await fetch(`${downloadLink}&key=${process.env.API_KEY}`);
                                const videoBlob = await videoResponse.blob();
                                const videoUrl = URL.createObjectURL(videoBlob);
                                setTranscript(prev => [...prev, { type: 'video', author: 'model', videoUrl, prompt }]);
                            } else { throw new Error("No download link in response."); }
                            sessionPromiseRef.current?.then(s => s.sendToolResponse({ functionResponses: { id: fc.id, name: fc.name, response: { result: `Displayed video for: ${prompt}` } } }));
                        } catch (vidErr: any) {
                            console.error("Video generation failed:", vidErr);
                            if (vidErr.message?.includes("Requested entity was not found")) {
                                setHasSelectedKey(false);
                                setError("Sua chave de API parece inválida. Por favor, selecione uma nova chave.");
                                stopConversation();
                            }
                            sessionPromiseRef.current?.then(s => s.sendToolResponse({ functionResponses: { id: fc.id, name: fc.name, response: { result: `Failed to show video for: ${prompt}` } } }));
                        } finally {
                            setIsGeneratingVideo(false);
                        }
                    }
                  }
                }

                if (message.serverContent?.turnComplete) {
                  const fullInput = currentInputTranscriptionRef.current.trim();
                  const fullOutput = currentOutputTranscriptionRef.current.trim();
                  if(fullInput){ setTranscript(prev => [...prev, { author: 'user', text: fullInput, type: 'text' }]); }
                  if(fullOutput){ setTranscript(prev => [...prev, { author: 'model', text: fullOutput, type: 'text' }]); }
                  currentInputTranscriptionRef.current = '';
                  currentOutputTranscriptionRef.current = '';
                }

                const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                if (base64Audio && outputAudioContextRef.current) {
                    const audioBuffer = await decodeAudioData(decode(base64Audio), outputAudioContextRef.current, 24000, 1);
                    const currentTime = outputAudioContextRef.current.currentTime;
                    nextStartTimeRef.current = Math.max(nextStartTimeRef.current, currentTime);
                    const source = outputAudioContextRef.current.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(outputAudioContextRef.current.destination);
                    source.addEventListener('ended', () => audioSourcesRef.current.delete(source));
                    source.start(nextStartTimeRef.current);
                    nextStartTimeRef.current += audioBuffer.duration;
                    audioSourcesRef.current.add(source);
                }
                 if (message.serverContent?.interrupted) {
                    audioSourcesRef.current.forEach(source => source.stop());
                    audioSourcesRef.current.clear();
                    nextStartTimeRef.current = 0;
                }
              },
              onerror: (e) => {
                console.error('API Error during session:', e);
              },
              onclose: () => {
                if (!isClosingRef.current) {
                  console.warn('Connection closed unexpectedly.');
                  setError('A conexão foi perdida. Por favor, inicie uma nova conversa.');
                  setStatus(Status.IDLE);
                  cleanUpAudio();
                }
              },
            },
          });
          sessionPromiseRef.current = sessionPromise;
          await sessionPromise;
        } catch (e: unknown) {
          console.error(`Connection attempt ${retryCount + 1} failed:`, e);
          cleanUpAudio();

          if (retryCount < MAX_RETRIES) {
              const delay = INITIAL_RETRY_DELAY_MS * Math.pow(2, retryCount);
              setError(`Serviço indisponível. Tentando novamente em ${delay / 1000}s...`);
              
              retryTimeoutRef.current = window.setTimeout(() => {
                  attemptConnection(retryCount + 1);
              }, delay);
          } else {
              setError(`Não foi possível conectar após ${MAX_RETRIES} tentativas. Por favor, verifique sua conexão e tente mais tarde.`);
              setStatus(Status.IDLE);
          }
        }
    };
    
    attemptConnection();
  }, [cleanUpAudio, stopConversation]);

  const handleToggleConversation = useCallback(() => {
    if (status === Status.IDLE) startConversation();
    else stopConversation();
  }, [status, startConversation, stopConversation]);

  const handleSelectKey = async () => {
    if((window as any).aistudio) {
        await (window as any).aistudio.openSelectKey();
        setHasSelectedKey(true);
    }
  };

  useEffect(() => {
    return () => { stopConversation(); };
  }, [stopConversation]);

  const renderTranscript = () => (
    <>
      {transcript.map((entry, index) => {
        if (entry.type === 'text') {
          return (
            <div key={index} className={`flex items-start gap-4 ${entry.author === 'user' ? 'justify-end' : ''}`}>
              {entry.author === 'model' && (<div className="w-8 h-8 rounded-full bg-indigo-500 flex-shrink-0 flex items-center justify-center font-bold text-sm">AI</div>)}
              <div className={`max-w-xs md:max-w-md lg:max-w-2xl p-4 rounded-2xl ${entry.author === 'user' ? 'bg-blue-600 rounded-br-none' : 'bg-gray-700 rounded-bl-none'}`}>
                <p className="text-white">{entry.text}</p>
              </div>
            </div>
          );
        }
        if (entry.type === 'image') {
          return (
            <div key={index} className="flex items-start gap-4 justify-start">
              <div className="w-8 h-8 rounded-full bg-indigo-500 flex-shrink-0 flex items-center justify-center font-bold text-sm">AI</div>
              <div className="max-w-xs md:max-w-md p-2 bg-gray-700 rounded-2xl rounded-bl-none">
                <img src={entry.imageUrl} alt={entry.prompt} className="rounded-lg w-full h-auto" />
              </div>
            </div>
          );
        }
        if (entry.type === 'pronunciation') {
          return (
             <div key={index} className="flex items-start gap-4 justify-start">
              <div className="w-8 h-8 rounded-full bg-indigo-500 flex-shrink-0 flex items-center justify-center font-bold text-sm">AI</div>
              <div className="max-w-xs md:max-w-md p-4 bg-gray-800 border border-green-500/50 rounded-2xl rounded-bl-none">
                <div className="flex items-baseline gap-3">
                    <span className="text-xl font-bold text-white">{entry.word}</span>
                    <span className="text-lg text-green-300">~ {entry.approximation}</span>
                </div>
                <p className="text-gray-300 mt-2 text-sm whitespace-pre-wrap">{entry.explanation}</p>
              </div>
            </div>
          )
        }
        if (entry.type === 'video') {
          return (
            <div key={index} className="flex items-start gap-4 justify-start">
              <div className="w-8 h-8 rounded-full bg-indigo-500 flex-shrink-0 flex items-center justify-center font-bold text-sm">AI</div>
              <div className="max-w-xs md:max-w-md p-2 bg-gray-700 rounded-2xl rounded-bl-none">
                <video src={entry.videoUrl} controls autoPlay muted loop className="rounded-lg w-full h-auto" />
              </div>
            </div>
          );
        }
        return null;
      })}
      {isGeneratingImage && (
        <div className="flex items-start gap-4 justify-start">
          <div className="w-8 h-8 rounded-full bg-indigo-500 flex-shrink-0 flex items-center justify-center font-bold text-sm">AI</div>
          <div className="p-4 bg-gray-700 rounded-2xl rounded-bl-none"><p className="text-white animate-pulse">Gerando imagem...</p></div>
        </div>
      )}
      {isGeneratingVideo && (
        <div className="flex items-start gap-4 justify-start">
          <div className="w-8 h-8 rounded-full bg-indigo-500 flex-shrink-0 flex items-center justify-center font-bold text-sm">AI</div>
          <div className="p-4 bg-gray-700 rounded-2xl rounded-bl-none"><p className="text-white animate-pulse">Gerando vídeo de articulação...</p></div>
        </div>
      )}
      <div ref={transcriptEndRef} />
    </>
  );

  if (!hasSelectedKey) {
    return (
        <div className="h-screen w-screen flex flex-col items-center justify-center bg-gray-900 font-sans text-white p-4 text-center">
            <h1 className="text-2xl font-bold mb-4">API Key Necessária</h1>
            <p className="max-w-md mb-6 text-gray-300">
                Para gerar vídeos de articulação, esta aplicação precisa de uma chave de API de um projeto Google Cloud com faturamento ativado.
            </p>
            <button
                onClick={handleSelectKey}
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition-colors"
            >
                Selecionar Chave de API
            </button>
            <p className="text-xs text-gray-500 mt-4">
                Saiba mais sobre <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noopener noreferrer" className="underline hover:text-blue-400">faturamento para a API Gemini</a>.
            </p>
        </div>
    );
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-gray-900 font-sans">
      <header className="p-4 border-b border-gray-700 text-center">
        <h1 className="text-2xl font-bold text-white">Ambiente de Aquisição de Inglês</h1>
        <div className="mt-2 text-sm text-gray-400 bg-gray-800/50 inline-block px-3 py-1 rounded-full">
            <span className="font-semibold">{phase.name}</span> | <span className="italic">Idade linguística: {phase.age}</span>
        </div>
      </header>
      
      <div className="flex-grow p-4 md:p-6 space-y-6 overflow-y-auto">
        {transcript.length === 0 && !isGeneratingImage && !isGeneratingVideo && !error && (
            <div className="flex h-full flex-col items-center justify-center text-center p-4">
            <div className="bg-gray-800 p-8 rounded-full mb-6"><MicIcon className="w-16 h-16 text-gray-500"/></div>
            <h2 className="text-xl font-semibold text-gray-300">Pronta para começar?</h2>
            <p className="text-gray-400 mt-2 max-w-sm">Pressione o microfone para iniciar sua imersão em inglês.</p>
            </div>
        )}
        {renderTranscript()}
      </div>

      {error && (<div className="p-4 text-center text-yellow-400 bg-yellow-900/50"><p><strong>Aviso:</strong> {error}</p></div>)}

      <footer className="p-4 flex flex-col items-center justify-center border-t border-gray-700">
        <div className="flex items-center space-x-4 mb-3">
          <div className={`w-3 h-3 rounded-full transition-colors duration-300 ${status === Status.LISTENING ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
          <p className="text-gray-400 text-sm">{status === Status.IDLE ? 'Desconectado' : 'Conectando...'}</p>
        </div>
        <button
          onClick={handleToggleConversation}
          className={`w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-offset-2 focus:ring-offset-gray-900 ${status === Status.LISTENING ? 'bg-red-600 hover:bg-red-700 focus:ring-red-500' : 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500'}`}
          aria-label={status === Status.IDLE ? "Iniciar conversa" : "Parar conversa"}>
          {status === Status.IDLE ? <MicIcon className="w-10 h-10 text-white" /> : <StopIcon className="w-10 h-10 text-white" />}
        </button>
      </footer>
    </div>
  );
};


// =================================================================
// RENDER LOGIC (from original index.tsx)
// =================================================================
const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
