interface Props {
  current?: number;
  total?: number;
  message?: string;
  indeterminate?: boolean;
  className?: string;
}

export default function ProgressLine({
  current = 0,
  total = 0,
  message,
  indeterminate = false,
  className = "",
}: Props) {
  const pct = total > 0 ? Math.min(100, (current / total) * 100) : 0;
  return (
    <div className={`space-y-1.5 ${className}`}>
      <div className="progress-bar">
        {indeterminate ? (
          <div
            className="h-full w-1/3 rounded-full bg-accent"
            style={{
              boxShadow: "0 0 12px rgba(124, 199, 255, 0.45)",
              animation: "indeterminate 1.4s ease-in-out infinite",
            }}
          />
        ) : (
          <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
        )}
      </div>
      {message && <p className="text-[11px] text-surface-400">{message}</p>}
      <style>{`@keyframes indeterminate { 0% { margin-left: -33%; } 100% { margin-left: 100%; } }`}</style>
    </div>
  );
}
