import type { ReactNode } from "react";

interface Props {
  label?: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  trailing?: ReactNode;
  className?: string;
  children: ReactNode;
}

export default function Field({ label, hint, error, trailing, className = "", children }: Props) {
  return (
    <div className={`space-y-1.5 ${className}`}>
      {(label || trailing) && (
        <div className="flex items-center justify-between">
          {label && (
            <label className="text-[12px] font-medium text-surface-300 uppercase tracking-wider">
              {label}
            </label>
          )}
          {trailing && <div className="text-[11px] text-surface-500">{trailing}</div>}
        </div>
      )}
      {children}
      {hint && !error && <p className="text-[11px] text-surface-500">{hint}</p>}
      {error && <p className="text-[11px] text-red-400">{error}</p>}
    </div>
  );
}
