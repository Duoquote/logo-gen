import type { HTMLAttributes, ReactNode } from "react";

type Tone = "default" | "soft" | "strong";

interface Props extends HTMLAttributes<HTMLElement> {
  tone?: Tone;
  as?: "section" | "div" | "article" | "aside";
  children: ReactNode;
}

const toneClass: Record<Tone, string> = {
  default: "glass-panel",
  soft: "glass-panel-soft",
  strong: "glass-panel-strong",
};

export default function GlassPanel({
  tone = "default",
  as: Tag = "section",
  className = "",
  children,
  ...rest
}: Props) {
  return (
    <Tag className={`${toneClass[tone]} ${className}`} {...rest}>
      {children}
    </Tag>
  );
}
