import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "icon";
type Size = "sm" | "md" | "lg";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  children?: ReactNode;
}

const base =
  "inline-flex items-center justify-center gap-2 font-medium rounded-field " +
  "transition-all duration-150 disabled:opacity-40 disabled:pointer-events-none " +
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40 " +
  "active:scale-[0.97]";

const variantClass: Record<Variant, string> = {
  primary:
    "bg-accent text-accent-ink font-semibold hover:bg-accent-hover hover:shadow-glow",
  secondary:
    "bg-glass border border-glass-border-strong text-surface-100 " +
    "hover:bg-glass-strong",
  ghost:
    "text-surface-400 hover:text-surface-100 hover:bg-glass",
  danger:
    "bg-red-500/80 text-white font-semibold hover:bg-red-500 hover:shadow-[0_0_24px_-4px_rgba(248,113,113,0.45)]",
  icon:
    "text-surface-300 hover:text-surface-100 hover:bg-glass",
};

const sizeClass: Record<Size, string> = {
  sm: "h-8 px-2.5 text-[13px]",
  md: "h-9 px-3.5 text-sm",
  lg: "h-11 px-5 text-[15px]",
};

const iconSize: Record<Size, string> = {
  sm: "w-7 h-7 px-0",
  md: "w-9 h-9 px-0",
  lg: "w-11 h-11 px-0",
};

export default function Button({
  variant = "primary",
  size = "md",
  className = "",
  children,
  ...rest
}: Props) {
  const sizeC = variant === "icon" ? iconSize[size] : sizeClass[size];
  const blur = variant === "secondary" ? " backdrop-blur-sm" : "";
  return (
    <button
      className={`${base} ${variantClass[variant]} ${sizeC}${blur} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}
