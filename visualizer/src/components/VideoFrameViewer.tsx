'use client';

import { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

// Type for image frames — supports both __image__-tagged dicts (RLM traces),
// KUAVi MCP format ({data, mime_type}), and sidecar file references ({_frame_file})
interface ImageFrame {
  __image__?: true;
  data: string;
  mime_type: string;
  timestamp?: string;
  /** Sidecar frame file name (e.g. "frame_0001.jpg") — used when data is loaded from API */
  _frame_file?: string;
}

// A segment groups frames with optional metadata
interface FrameSegment {
  label?: string;
  frames: ImageFrame[];
}

interface VideoFrameViewerProps {
  /** Flat list of frames or segmented groups */
  frames?: ImageFrame[];
  segments?: FrameSegment[];
  /** Max thumbnail width in pixels */
  thumbSize?: number;
  className?: string;
  /** Log stem for resolving sidecar frame file references */
  logStem?: string;
}

/** Check if a value matches an image frame pattern.
 *  Accepts __image__-tagged dicts, KUAVi MCP format ({data, mime_type}),
 *  and sidecar file references ({_frame_file, mime_type}). */
export function isImageFrame(value: unknown): value is ImageFrame {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  // Sidecar file reference (from experiment runner)
  if (typeof obj._frame_file === 'string' && typeof obj.mime_type === 'string') return true;
  if (typeof obj.data !== 'string' || typeof obj.mime_type !== 'string') return false;
  // Accept if __image__ tag is present OR if mime_type looks like an image
  return obj.__image__ === true || (obj.mime_type as string).startsWith('image/');
}

/**
 * Recursively extract all __image__ frames from an arbitrarily nested structure.
 * Handles: arrays, objects, message lists with content arrays, etc.
 */
export function extractImageFrames(data: unknown): ImageFrame[] {
  const frames: ImageFrame[] = [];

  function walk(node: unknown) {
    if (node === null || node === undefined) return;

    if (isImageFrame(node)) {
      frames.push(node);
      return;
    }

    if (Array.isArray(node)) {
      for (const item of node) walk(item);
      return;
    }

    if (typeof node === 'object') {
      for (const val of Object.values(node as Record<string, unknown>)) {
        walk(val);
      }
    }
  }

  walk(data);
  return frames;
}

/**
 * Check if a string contains what looks like base64 image data
 * (long alphanumeric runs typical of base64-encoded images).
 */
export function containsBase64ImageData(text: string): boolean {
  // Match base64 blocks ≥200 chars (a 320×240 JPEG is ~10k+ chars)
  return /[A-Za-z0-9+/]{200,}={0,2}/.test(text);
}

/**
 * Replace inline base64 image data in a string with a short placeholder.
 */
export function replaceBase64WithPlaceholder(text: string): string {
  return text.replace(
    /[A-Za-z0-9+/]{200,}={0,2}/g,
    '[image data]'
  );
}

/** Resolve image src — supports both inline base64 and sidecar API URLs */
function frameSrc(frame: ImageFrame, logStem?: string): string {
  if (frame.data) {
    return `data:${frame.mime_type};base64,${frame.data}`;
  }
  if (frame._frame_file && logStem) {
    return `/api/frames/${logStem}.frames/${frame._frame_file}`;
  }
  return '';
}

/** Render a single frame thumbnail */
function FrameThumb({ frame, size, index, logStem }: { frame: ImageFrame; size: number; index: number; logStem?: string }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const src = frameSrc(frame, logStem);

  if (!src) return null;

  return (
    <>
      <button
        onClick={() => setIsExpanded(true)}
        className="group relative rounded-md overflow-hidden border border-border hover:border-cyan-500/50 transition-all hover:shadow-md hover:shadow-cyan-500/10 focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={`Frame ${index + 1}`}
          width={size}
          height={Math.round(size * 0.75)}
          className="block object-cover bg-black"
          loading="lazy"
        />
        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/70 to-transparent px-1.5 py-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <span className="text-[9px] text-white font-mono">
            #{index + 1}{frame.timestamp ? ` · ${Number(frame.timestamp).toFixed(1)}s` : ''}
          </span>
        </div>
      </button>

      {/* Lightbox */}
      {isExpanded && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8 cursor-pointer"
          onClick={() => setIsExpanded(false)}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={`Frame ${index + 1} (expanded)`}
            className="max-w-full max-h-full rounded-lg shadow-2xl border border-white/10"
          />
          <span className="absolute top-4 right-4 text-white/60 text-sm">Click to close</span>
        </div>
      )}
    </>
  );
}

/** Filmstrip / grid viewer for video frames */
export function VideoFrameViewer({ frames, segments, thumbSize = 120, className, logStem }: VideoFrameViewerProps) {
  // Normalise into segments
  const resolvedSegments: FrameSegment[] = segments
    ? segments
    : frames
      ? [{ frames }]
      : [];

  const totalFrames = resolvedSegments.reduce((n, s) => n + s.frames.length, 0);

  if (totalFrames === 0) return null;

  return (
    <div className={cn('space-y-3', className)}>
      {resolvedSegments.map((segment, segIdx) => (
        <div key={`seg-${segment.label ?? segIdx}-${segment.frames.length}`}>
          {/* Segment header (only when there are multiple segments or a label) */}
          {(resolvedSegments.length > 1 || segment.label) && (
            <div className="flex items-center gap-2 mb-2">
              {segment.label && (
                <Badge variant="outline" className="text-[10px] font-mono">
                  {segment.label}
                </Badge>
              )}
              <span className="text-[10px] text-muted-foreground">
                {segment.frames.length} frame{segment.frames.length !== 1 ? 's' : ''}
              </span>
            </div>
          )}

          {/* Frame filmstrip */}
          <div className="flex flex-wrap gap-1.5">
            {segment.frames.map((frame, frameIdx) => {
              // Global index across all segments
              const globalIdx = resolvedSegments
                .slice(0, segIdx)
                .reduce((n, s) => n + s.frames.length, 0) + frameIdx;

              return (
                <FrameThumb
                  key={`frame-${segIdx}-${frameIdx}`}
                  frame={frame}
                  size={thumbSize}
                  index={globalIdx}
                  logStem={logStem}
                />
              );
            })}
          </div>
        </div>
      ))}

      {/* Summary badge */}
      <div className="flex items-center gap-2">
        <Badge className="bg-cyan-500/15 text-cyan-600 dark:text-cyan-400 border-cyan-500/30 text-[10px]">
          {totalFrames} video frame{totalFrames !== 1 ? 's' : ''}
        </Badge>
      </div>
    </div>
  );
}
