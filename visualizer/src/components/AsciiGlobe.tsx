'use client';

import { useEffect, useState } from 'react';

// RLM Architecture ASCII art inspired by the diagram
const RLM_SIMPLE = `
                    ╔══════════════════════════════════════════╗
  ┌──────────┐      ║            RLM (depth=0)                 ║      ┌──────────┐
  │  Prompt  │      ║  ┌────────────────────────────────────┐  ║      │  Answer  │
  │──────────│ ───► ║  │        Language Model (LM)         │  ║ ───► │──────────│
  │ context  │      ║  └─────────────────┬──────────────────┘  ║      │  FINAL() │
  └──────────┘      ║                   ↓ ↑                    ║      └──────────┘
                    ║  ┌─────────────────▼──────────────────┐  ║
                    ║  │       Environment (REPL)           │  ║
                    ║  │     context · llm_query()          │  ║
                    ║  └──────────┬────────────┬────────────┘  ║
                    ╚═════════════│════════════│═══════════════╝
                                  │            │
                         ┌────────▼────┐  ┌────▼────────┐
                         │ llm_query() │  │ llm_query() │
                         └────────┬────┘  └────┬────────┘
                                  │            │
                         ╔════════▼════╗  ╔════▼════════╗
                         ║ RLM (d=1)   ║  ║ RLM (d=1)   ║
                         ║  LM ↔ REPL  ║  ║  LM ↔ REPL  ║
                         ╚═════════════╝  ╚═════════════╝
`;

export function AsciiRLM() {
  const [pulse, setPulse] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setPulse(p => (p + 1) % 4);
    }, 600);
    return () => clearInterval(interval);
  }, []);

  // Colorize the ASCII art
  const colorize = (text: string) => {
    return text.split('\n').map((line, lineIdx) => (
      <div key={`line-${lineIdx}`} className="whitespace-pre">
        {line.split('').map((char, charIdx) => {
          const key = `char-${lineIdx}-${charIdx}`;
          
          // Box drawing characters - dim
          if ('┌┐└┘├┤┬┴┼─│╔╗╚╝║═'.includes(char)) {
            return <span key={key} className="text-muted-foreground/50">{char}</span>;
          }
          // Arrows - primary color
          if ('▼▲↓↑→←'.includes(char)) {
            const isPulsing = (lineIdx + charIdx + pulse) % 4 === 0;
            return (
              <span 
                key={key} 
                className={isPulsing ? 'text-primary' : 'text-primary/60'}
              >
                {char}
              </span>
            );
          }
          // Keywords
          if (line.includes('RLM') && char !== ' ') {
            if ('RLM'.includes(char)) {
              return <span key={key} className="text-primary font-bold">{char}</span>;
            }
          }
          if (line.includes('Prompt') || line.includes('Response') || line.includes('Answer')) {
            if (!'[]│─'.includes(char) && char !== ' ') {
              return <span key={key} className="text-amber-600 dark:text-amber-400">{char}</span>;
            }
          }
          if (line.includes('Language Model') || line.includes('LM')) {
            if (!'[]│─┌┐└┘'.includes(char) && char !== ' ') {
              return <span key={key} className="text-sky-600 dark:text-sky-400">{char}</span>;
            }
          }
          if (line.includes('REPL') || line.includes('Environment') || line.includes('context') || line.includes('llm_query')) {
            if (!'[]│─┌┐└┘'.includes(char) && char !== ' ') {
              return <span key={key} className="text-emerald-600 dark:text-emerald-400">{char}</span>;
            }
          }
          if (line.includes('depth=')) {
            if (!'()'.includes(char) && char !== ' ') {
              return <span key={key} className="text-muted-foreground">{char}</span>;
            }
          }
          // Default
          return <span key={key} className="text-muted-foreground/70">{char}</span>;
        })}
      </div>
    ));
  };

  return (
    <div className="font-mono text-[10px] leading-[1.3] select-none">
      <pre>{colorize(RLM_SIMPLE)}</pre>
    </div>
  );
}

// KUAVi Architecture ASCII art — agentic video analysis pipeline
const KUAVI_DIAGRAM = `
  ┌──────────┐      ┌──────────────────────────────────────────┐
  │  Video   │      │           Indexing Pipeline               │
  │──────────│ ───► │  V-JEPA 2  │  SigLIP2  │  Whisper ASR   │
  │   .mp4   │      └────────────────────┬─────────────────────┘
  └──────────┘                           │
                               ┌─────────▼─────────┐
                               │    Video Index     │
                               │ segments · embeds  │
                               └─────────┬─────────┘
                                         ▼
  ┌──────────┐      ╔══════════════════════════════════════════╗      ┌──────────┐
  │ Question │      ║           LLM Agent (KUAVi)              ║      │  Answer  │
  │──────────│ ───► ║  ┌────────────────────────────────────┐  ║ ───► │──────────│
  │  query   │      ║  │       Language Model (LM)          │  ║      │  FINAL() │
  └──────────┘      ║  └─────────────────┬──────────────────┘  ║      └──────────┘
                    ║                   ↓ ↑                    ║
                    ║  ┌─────────────────▼──────────────────┐  ║
                    ║  │         MCP Tools (18)             │  ║
                    ║  │ search · frames · pixel · transcript║
                    ║  └────────────────────────────────────┘  ║
                    ╚══════════════════════════════════════════╝
`;

export function AsciiKUAVi() {
  const [pulse, setPulse] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setPulse(p => (p + 1) % 4);
    }, 600);
    return () => clearInterval(interval);
  }, []);

  const colorize = (text: string) => {
    return text.split('\n').map((line, lineIdx) => (
      <div key={`line-${lineIdx}`} className="whitespace-pre">
        {line.split('').map((char, charIdx) => {
          const key = `char-${lineIdx}-${charIdx}`;

          // Box drawing characters - dim
          if ('┌┐└┘├┤┬┴┼─│╔╗╚╝║═'.includes(char)) {
            return <span key={key} className="text-muted-foreground/50">{char}</span>;
          }
          // Arrows - pulsing primary
          if ('▼▲↓↑→←◄►◈'.includes(char)) {
            const isPulsing = (lineIdx + charIdx + pulse) % 4 === 0;
            return (
              <span
                key={key}
                className={isPulsing ? 'text-primary' : 'text-primary/60'}
              >
                {char}
              </span>
            );
          }
          // Model names — violet bold (check before generic Video/Index)
          if (line.includes('V-JEPA') || line.includes('SigLIP2') || line.includes('Whisper')) {
            if (!'[]│─┌┐└┘'.includes(char) && char !== ' ') {
              return <span key={key} className="text-violet-600 dark:text-violet-400 font-bold">{char}</span>;
            }
          }
          // Video / Index keywords — violet
          if (line.includes('Video') || line.includes('Index') || line.includes('Indexing') || line.includes('segments') || line.includes('embeds')) {
            if (!'[]│─┌┐└┘·'.includes(char) && char !== ' ') {
              return <span key={key} className="text-violet-600 dark:text-violet-400">{char}</span>;
            }
          }
          // LLM / Agent / KUAVi — sky
          if (line.includes('LLM') || line.includes('Agent') || line.includes('KUAVi') || line.includes('Language Model')) {
            if (!'[]│─┌┐└┘()'.includes(char) && char !== ' ') {
              return <span key={key} className="text-sky-600 dark:text-sky-400">{char}</span>;
            }
          }
          // MCP Tools — emerald
          if (line.includes('MCP') || line.includes('Tools') || line.includes('search') || line.includes('frames') || line.includes('pixel') || line.includes('transcript')) {
            if (!'[]│─┌┐└┘()'.includes(char) && char !== ' ') {
              return <span key={key} className="text-emerald-600 dark:text-emerald-400">{char}</span>;
            }
          }
          // Input/Output — amber
          if (line.includes('Question') || line.includes('Answer') || line.includes('query') || line.includes('FINAL') || line.includes('.mp4')) {
            if (!'[]│─┌┐└┘'.includes(char) && char !== ' ') {
              return <span key={key} className="text-amber-600 dark:text-amber-400">{char}</span>;
            }
          }
          // Default
          return <span key={key} className="text-muted-foreground/70">{char}</span>;
        })}
      </div>
    ));
  };

  return (
    <div className="font-mono text-[10px] leading-[1.3] select-none">
      <pre>{colorize(KUAVI_DIAGRAM)}</pre>
    </div>
  );
}

// Compact inline diagram for header
export function AsciiRLMInline() {
  return (
    <div className="font-mono text-[9px] leading-tight select-none text-muted-foreground">
      <span className="text-primary">Prompt</span>
      <span> → </span>
      <span className="text-emerald-600 dark:text-emerald-400">[LM ↔ REPL]</span>
      <span> → </span>
      <span className="text-amber-600 dark:text-amber-400">Answer</span>
    </div>
  );
}
