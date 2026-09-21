interface Option<T extends string> {
  value: T;
  label: string;
}

interface Props<T extends string> {
  value: T;
  options: Option<T>[];
  onChange: (value: T) => void;
  size?: "sm" | "md";
  className?: string;
}

export default function SegmentedControl<T extends string>({
  value,
  options,
  onChange,
  size = "md",
  className = "",
}: Props<T>) {
  const h = size === "sm" ? "h-8" : "h-9";
  const px = size === "sm" ? "px-3 text-[13px]" : "px-4 text-sm";
  return (
    <div
      role="tablist"
      className={`inline-flex items-center p-1 ${h} bg-surface-800/60 border border-glass-border-strong rounded-pill ${className}`}
    >
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            role="tab"
            aria-selected={active}
            type="button"
            onClick={() => onChange(opt.value)}
            className={`
              relative rounded-pill ${px} font-medium transition-all duration-150
              ${
                active
                  ? "bg-accent text-accent-ink shadow-glass-sm"
                  : "text-surface-300 hover:text-surface-100"
              }
            `}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
