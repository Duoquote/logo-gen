# Logo Design Principles for AI Logo Generation

Research compiled: 2026-03-27

This document encodes logo design principles, brand psychology, and quality criteria
to guide the AI logo generator's prompt construction, output evaluation, and
recommendation engine.

---

## 1. Core Design Principles

Every generated logo must be evaluated against these five foundational principles:

### 1.1 Simplicity
- The most recognizable logos in the world are deceptively simple (Nike swoosh, Apple, Target).
- Strip away every element that does not serve a purpose. If removing an element does not reduce meaning, it should not be there.
- Limit to 2-3 colors maximum for the primary mark.
- Limit to 1-2 typefaces.
- Avoid gradients, shadows, and 3D effects in the primary mark (they fail at small sizes).
- **AI prompt guidance**: Always include "simple", "clean", "minimal detail" as base descriptors. Penalize outputs with excessive ornamentation.

### 1.2 Scalability
- A logo must be legible and recognizable from a 16x16 favicon to a billboard.
- Vector-based output is mandatory; raster outputs are only acceptable as previews.
- Fine details, thin strokes, and small text break at small sizes.
- Test every output at: 16px (favicon), 48px (app icon), 200px (web header), 1000px+ (print).
- **AI prompt guidance**: Specify "works at small sizes", "bold clear shapes". Include scalability as a mandatory evaluation gate.

### 1.3 Memorability
- A memorable logo can be sketched from memory after a brief viewing.
- Achieved through distinctive shape, unexpected color pairing, or clever use of negative space.
- Avoid generic clip-art-like imagery (globes, arrows, generic people silhouettes).
- **AI prompt guidance**: Request "distinctive", "unique silhouette". Score outputs on shape uniqueness via silhouette comparison.

### 1.4 Versatility
- Must work on light backgrounds, dark backgrounds, and photographic backgrounds.
- Must work in full color, single color, and black-and-white.
- Must work horizontally, vertically (stacked), and as an icon-only mark.
- Deliver every logo as a system: full lockup, icon only, wordmark only, monochrome, reversed.
- **AI prompt guidance**: Generate and test multiple background contexts. Reject logos that rely on color alone for recognition.

### 1.5 Timelessness
- A logo should aim for a 10-20 year lifespan minimum.
- Avoid hyper-trendy effects (current year's fad gradients, overly stylized 3D).
- Draw from enduring design archetypes rather than fleeting aesthetics.
- Classic geometric shapes, clean typography, and restrained color age best.
- **AI prompt guidance**: Include "timeless", "enduring" in prompts. Down-weight outputs that rely heavily on trend-specific effects.

---

## 2. Logo Types

### 2.1 Wordmark (Logotype)
**Definition**: The full brand name rendered in a distinctive typeface.
**Examples**: Google, Coca-Cola, FedEx, Visa, Disney.

**Characteristics**:
- Typography IS the logo; no separate icon.
- Works best with short, distinctive names (1-2 words).
- Builds direct name recognition.
- Prints cleanly across signage, packaging, and web headers.

**When to recommend**:
- Brand name is short (ideally 1 word, max 2 words).
- The name itself is distinctive or invented (e.g., "Spotify", "Etsy").
- The brand is new and needs to establish name awareness.
- The brand will appear primarily in text-heavy contexts (editorial, SaaS, professional services).
- The brand wants a clean, modern, typography-forward identity.

**When to avoid**:
- The name is long (3+ words) or generic.
- The brand needs a standalone icon for app icons, social avatars, or favicons.

### 2.2 Lettermark (Monogram)
**Definition**: Initials or abbreviated letters designed as a cohesive mark.
**Examples**: IBM, HBO, CNN, NASA, LV (Louis Vuitton).

**Characteristics**:
- Condenses a long name into a compact, recognizable symbol.
- Works well at small sizes (app icons, social avatars, stamps, embroidery).
- Requires strong typographic design to feel distinctive rather than generic.

**When to recommend**:
- Brand name is long (3+ words) and the initials are memorable.
- The brand operates across many physical touchpoints (business cards, uniforms, products).
- The brand needs a compact mark for digital-first environments (apps, social media).
- The brand name is already commonly referred to by its initials.

**When to avoid**:
- The initials are forgettable or spell something undesirable.
- The brand is unknown and the initials carry no recognition.

### 2.3 Symbol / Pictorial Mark
**Definition**: A recognizable icon or graphic that represents the brand without text.
**Examples**: Apple (apple), Twitter/X (bird), Target (bullseye), Shell (shell).

**Characteristics**:
- Highly compact and versatile across sizes and media.
- Transcends language barriers -- works globally without translation.
- Requires significant brand awareness to stand alone.

**When to recommend**:
- The brand is already well-established or planning major awareness campaigns.
- The brand operates internationally across language barriers.
- There is a clear, meaningful visual metaphor that connects to the brand's core offering.
- The brand needs a strong app icon or social media avatar.

**When to avoid**:
- The brand is new and unknown (a symbol alone will not build name recognition).
- There is no clear visual metaphor -- forcing one produces generic results.

### 2.4 Combination Mark
**Definition**: A wordmark or lettermark paired with a symbol, icon, or graphic element.
**Examples**: Adidas, Burger King, Lacoste, Doritos, Amazon.

**Characteristics**:
- The most versatile logo type. Over 60% of Fortune 500 companies use combination marks.
- Can be separated: the icon and wordmark can work independently as the brand gains recognition.
- Layouts: side-by-side, stacked, or integrated.

**When to recommend**:
- Default recommendation for most new brands. Offers the best balance of name recognition and visual identity.
- The brand wants flexibility to use the icon alone in some contexts and the full lockup in others.
- The brand operates across both digital and physical touchpoints.

**When to avoid**:
- Rarely a bad choice, but can feel cluttered if the icon and text do not harmonize.
- If extreme simplicity is the goal, a wordmark may be cleaner.

### 2.5 Emblem
**Definition**: Text enclosed within or integrated into a symbol, badge, seal, or crest.
**Examples**: Starbucks, Harley-Davidson, NFL, Harvard, BMW.

**Characteristics**:
- Conveys heritage, authority, tradition, and craftsmanship.
- Compact and unified -- the text and symbol are inseparable.
- Can struggle at very small sizes due to fine detail.

**When to recommend**:
- The brand wants to communicate heritage, prestige, or authority (universities, government, luxury, automotive, craft beverages).
- The logo will appear primarily on physical goods (labels, packaging, merchandise).
- The brand identity leans toward classic or traditional.

**When to avoid**:
- Digital-first brands that need extreme scalability (emblems often fail at 16px).
- Brands wanting a modern, minimal aesthetic.

### 2.6 Abstract Mark
**Definition**: A geometric or organic shape that represents the brand conceptually rather than literally.
**Examples**: Nike (swoosh), Pepsi, Airbnb, Adidas (three stripes), BP.

**Characteristics**:
- Completely unique -- no risk of overlapping with another brand's literal imagery.
- Can encode multiple meanings simultaneously.
- Requires deliberate brand-building to establish meaning.

**When to recommend**:
- The brand wants to stand out from competitors using similar literal imagery.
- The brand's value proposition is conceptual or multifaceted.
- The brand wants complete trademark-ability without literal conflict.
- Technology, consulting, or multi-product companies where no single image captures the brand.

**When to avoid**:
- The brand has a strong, obvious visual metaphor that would be wasted.
- The brand is small and local -- abstract marks need investment to build meaning.

### 2.7 Mascot
**Definition**: A illustrated character that represents the brand.
**Examples**: KFC (Colonel Sanders), Michelin (Bibendum), Mailchimp (Freddie), Pringles.

**Characteristics**:
- Creates a strong emotional and personal connection.
- Works exceptionally well for family-oriented, food, sports, and entertainment brands.
- Naturally suited to storytelling, animation, and social media.
- Can feel less professional for serious B2B contexts.

**When to recommend**:
- Target audience includes families, children, or casual consumers.
- The brand wants to convey friendliness, approachability, and personality.
- Food, beverage, sports teams, gaming, or entertainment brands.
- The brand will heavily use social media and content marketing where a character can tell stories.

**When to avoid**:
- Professional services, finance, law, healthcare, or enterprise B2B.
- The brand wants a minimalist or serious tone.

---

## 3. Logo Type Recommendation Decision Tree

```
START
  |
  v
Is the brand name short (1-2 words) and distinctive?
  |-- YES --> Does the brand need a standalone icon? (app, favicon, avatar)
  |             |-- YES --> COMBINATION MARK
  |             |-- NO  --> WORDMARK
  |
  |-- NO (3+ words) --> Are the initials memorable?
                          |-- YES --> LETTERMARK (or COMBINATION with lettermark)
                          |-- NO  --> Shorten the name? Or use COMBINATION MARK

MODIFIERS (apply after primary type selection):
  - Brand conveys heritage/authority? --> Consider EMBLEM variant
  - Brand targets families/entertainment? --> Consider MASCOT variant
  - Brand is conceptual/multi-product? --> Consider ABSTRACT MARK
  - Brand has obvious visual metaphor? --> Consider SYMBOL or PICTORIAL MARK
  - Brand is new/unknown? --> Prefer types that include the name (wordmark, combination)
  - Brand is well-established? --> Can use symbol-only approaches
```

---

## 4. 2025-2026 Logo Design Trends

### 4.1 Adaptive Logo Systems
- Logos are no longer single, fixed marks. They are built as systems that adapt across platforms, formats, and environments.
- A logo may change color, shape, or composition based on context: tiny app icon vs. large billboard vs. dark mode vs. AR environment.
- **Implementation**: Generate logo variants at multiple resolutions and orientations. Produce responsive lockups (full, compact, icon-only).

### 4.2 Typography-Led Branding
- The defining trend of 2026: typography as the primary brand signal, with icons becoming secondary.
- Custom or modified typefaces that move, stretch, and react -- especially in motion-first environments.
- Brands rediscovering type as a storytelling tool: playful, rebellious, or distinctly human.
- **Implementation**: Invest heavily in wordmark quality. Offer custom-feel typography options. Score typographic distinctiveness.

### 4.3 Warm Minimalism (Neo-Minimalism / Minimalism 3.0)
- Minimalism remains dominant but evolves to feel more human and connected.
- Thick strokes, oversized lettering, high-contrast color pairings.
- Subtle character, warmth, and nuance without sacrificing clarity.
- Controlled imperfection adds personality -- overly perfect logos struggle to earn trust.
- **Implementation**: Offer "warm minimal" as a style option. Introduce slight organic irregularities in stroke weight or letter spacing.

### 4.4 Motion-First Design
- Logos designed with motion rules from the start, not retrofitted.
- Subtle animation helps logos feel alive in interface-driven environments.
- Motion captures attention without visual excess.
- **Implementation**: Define animation keyframes for generated logos. Provide CSS/SVG animation presets (fade-in, draw-on, morph).

### 4.5 Human Touch and Imperfection
- Brands that feel overly polished or machine-generated struggle to earn trust.
- Hand-drawn elements, organic shapes, and deliberate imperfection signal authenticity.
- **Implementation**: Offer "hand-crafted" style variants. Introduce subtle stroke variation, organic curves, or hand-lettering options.

### 4.6 Bold Color and High Contrast
- Moving away from muted startup pastels toward bolder, more confident palettes.
- High contrast pairings for accessibility and impact.
- Monochrome with a single accent color gaining popularity.
- **Implementation**: Default to high-contrast palettes. Validate WCAG contrast ratios. Offer bold accent color options.

### 4.7 Nostalgic and Retro Revival
- Y2K, 70s, and 90s aesthetics appearing in logo design.
- Retro type treatments, arched text, vintage badge formats.
- Used selectively to convey personality rather than as the dominant aesthetic.
- **Implementation**: Offer retro/vintage as a style category with era-specific presets.

### 4.8 AI-Aware Design
- As AI-generated logos become common, brands seek ways to signal intentionality and craftsmanship.
- Human-AI collaboration producing more expressive, dynamic marks.
- **Implementation**: Use AI for rapid iteration while providing human-centric refinement controls.

---

## 5. Common Logo Design Mistakes

These must be encoded as negative constraints in AI generation and as penalty criteria in quality scoring.

### 5.1 Overcomplexity
- **Mistake**: Too many elements, colors, fonts, or concepts crammed into one mark.
- **Consequence**: Illegible at small sizes, hard to remember, expensive to reproduce.
- **AI guard**: Limit generated logos to max 3 colors, 2 fonts, 1 primary concept. Penalize visual density above threshold.

### 5.2 Poor Scalability
- **Mistake**: Fine details, thin strokes, small text that break at small sizes.
- **Consequence**: Blurry favicons, illegible app icons, broken print at business-card scale.
- **AI guard**: Test all outputs at 16px, 48px, 200px. Reject logos where elements merge or disappear below 48px.

### 5.3 Mismatched Typography
- **Mistake**: Choosing fonts that contradict the brand's personality or using 3+ typefaces.
- **Consequence**: Confused brand message, amateurish appearance.
- **AI guard**: Map brand personality keywords to font categories. Restrict to 1-2 fonts. Validate font-mood alignment.

### 5.4 Color Harmony Violations
- **Mistake**: Colors with clashing saturations, hues, or insufficient contrast.
- **Consequence**: Visual discomfort, accessibility failure, unprofessional appearance.
- **AI guard**: Validate color combinations against established harmony rules (complementary, analogous, triadic). Enforce minimum contrast ratios.

### 5.5 Trend Dependency
- **Mistake**: Building the entire identity on a current trend (e.g., extreme gradients, 3D chrome).
- **Consequence**: Logo looks dated within 2-3 years.
- **AI guard**: Flag outputs that rely exclusively on trend-specific effects. Offer trend-influenced variants alongside a timeless base version.

### 5.6 Lack of Uniqueness
- **Mistake**: Generic shapes, stock-icon-like imagery, or unintentional similarity to existing logos.
- **Consequence**: Brand confusion, legal risk, failure to differentiate.
- **AI guard**: Compare generated silhouettes against a reference database of existing logos. Flag high-similarity matches. Penalize generic archetypes (globe + swoosh, generic person, basic gear).

### 5.7 Scale Mismatch
- **Mistake**: Disproportionate elements -- a huge icon with tiny unreadable text, or vice versa.
- **Consequence**: Visual imbalance, poor hierarchy.
- **AI guard**: Enforce minimum text-to-icon ratio. Validate visual weight distribution.

### 5.8 No Monochrome Fallback
- **Mistake**: Logo that only works in full color and becomes unrecognizable in black-and-white.
- **Consequence**: Fails on fax, newspaper, engraving, single-color printing.
- **AI guard**: Generate and score a monochrome version of every logo. Reject if monochrome version loses key features.

### 5.9 Inappropriate Tone
- **Mistake**: Playful mascot for a law firm, or a rigid emblem for a children's brand.
- **Consequence**: Audience disconnect, undermined credibility.
- **AI guard**: Map industry + audience inputs to appropriate style constraints. Hard-block mismatched type recommendations.

### 5.10 Plagiarism / Unintentional Copying
- **Mistake**: Output too similar to an existing well-known logo.
- **Consequence**: Legal risk, brand confusion, reputational damage.
- **AI guard**: Implement similarity detection against top 10,000 logos. Require minimum differentiation threshold.

---

## 6. Encoding Principles into AI Prompts and Evaluation

### 6.1 Prompt Construction Framework

Every AI logo generation prompt should be assembled from these layers:

```
LAYER 1 - Base Quality (always included):
  "Simple, clean, professional logo design. Bold clear shapes.
   Works at small sizes. Vector-style. High contrast.
   Single concept. Minimal detail."

LAYER 2 - Logo Type Specification:
  "[wordmark/lettermark/symbol/combination/emblem/abstract/mascot] logo"

LAYER 3 - Brand Identity:
  "For a [industry] brand called [name].
   Brand personality: [2-3 adjectives from brand input].
   Target audience: [audience description]."

LAYER 4 - Style Direction:
  "Style: [modern minimal / classic elegant / bold playful / etc.].
   Color palette: [specific colors or mood-based guidance].
   Typography: [serif/sans-serif/display + weight guidance]."

LAYER 5 - Negative Constraints (always included):
  "No photorealistic elements. No gradients unless specified.
   No more than 3 colors. No clip art. No generic stock imagery.
   No tiny text. No fine details that break at small sizes."

LAYER 6 - Technical Specifications:
  "Centered composition. Clean background. Suitable for
   vector conversion. Sharp edges. Clear silhouette."
```

### 6.2 Prompt Optimization Best Practices

1. **Be concrete, not abstract**: "A strong oak tree with clean geometric branches" beats "something that conveys growth and strength."
2. **Use visual anchors**: Specify shapes, compositions, and spatial relationships rather than abstract qualities.
3. **Avoid contradictions**: "Minimalist but detailed" or "playful but corporate" confuse the model.
4. **Prioritize constraints**: List the most important requirements first; models weight early tokens more heavily.
5. **Iterate in stages**: Stage 1: concept/symbol only. Stage 2: add typography. Stage 3: refine color. Stage 4: test variants.
6. **Limit scope per prompt**: One concept per generation. Avoid "give me 5 different styles" -- generate them separately with distinct prompts.

### 6.3 Evaluation Pipeline

```
GENERATION --> TECHNICAL FILTER --> QUALITY SCORING --> DIVERSITY CHECK --> PRESENTATION

Technical Filter (binary pass/fail):
  - Is the output a logo (not a scene, photograph, or illustration)?
  - Is the background clean / removable?
  - Is text legible (if text was requested)?
  - Does it survive downscaling to 48px without major loss?

Quality Scoring (see Section 7):
  - Scored 0-100 across multiple dimensions.
  - Minimum threshold: 60/100 to present to user.

Diversity Check:
  - Among batch outputs, ensure visual diversity.
  - Reject near-duplicates (>90% structural similarity).
  - Ensure at least one option from each requested style variant.
```

---

## 7. Logo Quality Scoring Criteria

Automated evaluation rubric for scoring generated logos on a 0-100 scale.

### 7.1 Scoring Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Simplicity | 15% | Fewer elements = higher score. Penalize visual clutter, excess colors, multiple competing focal points. |
| Scalability | 15% | Score based on legibility test at 16px, 48px, 200px, 1000px. All four must pass for full marks. |
| Distinctiveness | 15% | Silhouette uniqueness score. Compare against generic archetype database. Penalize cliches. |
| Relevance | 15% | How well does the logo match the stated brand personality, industry, and audience? Scored via prompt-output alignment. |
| Typography Quality | 10% | Legibility, appropriate font choice, proper spacing, hierarchy. N/A for symbol-only marks. |
| Color Effectiveness | 10% | Harmony, contrast ratio, emotional alignment with brand, monochrome viability. |
| Composition | 10% | Visual balance, proper use of space, clear focal point, professional alignment. |
| Versatility | 10% | Works on light/dark backgrounds, works in monochrome, separable icon+text components. |

### 7.2 Scoring Rubric Detail

**Simplicity (0-15)**
- 13-15: Single concept, 1-2 colors, clean and immediately readable.
- 9-12: Clean design with minor unnecessary detail.
- 5-8: Multiple competing elements or 4+ colors.
- 0-4: Cluttered, chaotic, or illustration-like rather than logo-like.

**Scalability (0-15)**
- 13-15: Fully legible and recognizable at 16px favicon size.
- 9-12: Works at 48px but loses detail at 16px.
- 5-8: Requires 200px+ to be legible.
- 0-4: Even at large sizes, elements are confused or overlapping.

**Distinctiveness (0-15)**
- 13-15: Unique silhouette, would not be confused with any well-known logo.
- 9-12: Somewhat distinctive but uses common structural patterns.
- 5-8: Generic feeling, could belong to many different brands.
- 0-4: Clip-art-like or directly resembles an existing well-known logo.

**Relevance (0-15)**
- 13-15: Immediately communicates the brand's industry and personality.
- 9-12: Generally appropriate, minor disconnect.
- 5-8: Vaguely relevant but could apply to many industries.
- 0-4: Contradicts the brand's personality or industry (e.g., playful for funerals).

**Typography Quality (0-10)**
- 9-10: Distinctive, legible, perfectly matched to brand personality.
- 6-8: Appropriate and clean, but not distinctive.
- 3-5: Legible but poorly matched or poorly spaced.
- 0-2: Illegible, broken, or mismatched.

**Color Effectiveness (0-10)**
- 9-10: Harmonious palette, strong contrast, emotionally aligned, works in mono.
- 6-8: Good palette, minor harmony or contrast issues.
- 3-5: Colors clash or fail contrast requirements.
- 0-2: Actively unpleasant or completely misaligned with brand.

**Composition (0-10)**
- 9-10: Perfectly balanced, clear hierarchy, professional.
- 6-8: Good composition, minor alignment or spacing issues.
- 3-5: Unbalanced or poorly centered.
- 0-2: Chaotic layout, no clear focal point.

**Versatility (0-10)**
- 9-10: Works across all contexts (light/dark/mono/icon/full).
- 6-8: Works in most contexts, struggles in one.
- 3-5: Requires full color on light background to work.
- 0-2: Only works in one very specific context.

### 7.3 Automated Scoring Implementation Notes

- **Simplicity**: Count distinct colors (via clustering), count distinct shapes (via contour detection), measure visual density (ink-to-whitespace ratio).
- **Scalability**: Render at target sizes, compute structural similarity (SSIM) between full-size and downscaled versions. Below SSIM threshold = fail.
- **Distinctiveness**: Extract edge/silhouette features, compare via cosine similarity against a reference database of 10K+ logos. Flag if similarity > 0.85.
- **Relevance**: Use CLIP or similar vision-language model to score alignment between the generated image and the brand description prompt.
- **Typography**: OCR the text region, validate spelling, measure character spacing consistency, classify font style and compare to brand-appropriate styles.
- **Color**: Extract dominant colors via k-means, validate against color harmony rules, compute WCAG contrast ratios, compare emotional associations to brand keywords.
- **Composition**: Analyze center of mass, symmetry score, rule-of-thirds alignment, whitespace distribution.
- **Versatility**: Render on white, black, and mid-gray backgrounds. Generate monochrome version. Score recognition across all variants.

---

## 8. Brand Psychology and Color Theory

### 8.1 Color Psychology Reference

| Color | Emotions / Associations | Best For Industries | Avoid When |
|-------|------------------------|--------------------|-----------  |
| **Blue** | Trust, security, intelligence, calm, professionalism | Finance, tech, healthcare, corporate, insurance | Food (suppresses appetite), children's brands |
| **Red** | Energy, passion, urgency, excitement, power | Food, entertainment, sports, retail, sales | Calm/wellness brands, financial security |
| **Green** | Growth, health, nature, freshness, wealth | Eco/sustainability, health, finance, organic food | Tech (unless eco-tech), luxury fashion |
| **Yellow** | Optimism, warmth, attention, happiness, youth | Food, children, retail, creative, budget brands | Luxury, premium, corporate seriousness |
| **Orange** | Friendly, energetic, confident, affordable, playful | Food, fitness, youth brands, ecommerce, SaaS | Luxury, premium, healthcare, finance |
| **Purple** | Luxury, creativity, wisdom, nobility, mystery | Beauty, luxury, creative, spiritual, education | Budget brands, industrial, agriculture |
| **Black** | Sophistication, luxury, power, elegance, authority | Fashion, luxury, tech, automotive, editorial | Children's brands, eco/nature, budget |
| **White** | Purity, simplicity, cleanliness, modern, space | Healthcare, tech, minimalist brands, wellness | N/A (typically used as a supporting color) |
| **Pink** | Playful, feminine, compassionate, romantic, sweet | Beauty, fashion, dating, sweets, feminine products | Heavy industry, B2B enterprise, finance |
| **Brown** | Earthy, reliable, rustic, warm, traditional | Food/coffee, outdoor, craft, heritage brands | Tech, modern/futuristic brands |

### 8.2 Color Combination Strategies

- **Monochromatic**: One hue, varying lightness/saturation. Safe, elegant, easy to execute. Risk: monotony.
- **Complementary**: Opposite colors on the color wheel (blue + orange). High contrast and energy. Risk: can be jarring if not balanced.
- **Analogous**: Adjacent colors on the wheel (blue + teal + green). Harmonious and natural. Risk: low contrast.
- **Triadic**: Three evenly spaced colors (red + yellow + blue). Vibrant and balanced. Risk: can feel childish or busy.
- **Split-complementary**: Base color + two adjacent to its complement. High contrast with more nuance than pure complementary.
- **Neutral + Accent**: Black/white/gray base with one bold accent color. Modern, flexible, high impact. Increasingly popular in 2025-2026.

### 8.3 Shape Psychology

| Shape | Associations | Best For |
|-------|-------------|----------|
| **Circles** | Unity, community, wholeness, protection, femininity, continuity | Social platforms, non-profits, wellness, food, global brands |
| **Squares/Rectangles** | Stability, reliability, order, strength, professionalism | Finance, construction, tech, legal, corporate |
| **Triangles (up)** | Power, progression, ambition, hierarchy, energy | Tech, fitness, innovation, leadership |
| **Triangles (down)** | Femininity, creativity, mysticism | Fashion, art, spiritual brands |
| **Horizontal lines** | Calm, community, tranquility | Wellness, hospitality, landscape |
| **Vertical lines** | Strength, authority, growth | Finance, government, luxury |
| **Organic/Curves** | Movement, warmth, creativity, nature | Food, beauty, creative, eco |
| **Spirals** | Growth, evolution, creativity, hypnotic | Creative, tech, education |

### 8.4 Typography Psychology

| Font Category | Personality | Best For |
|--------------|-------------|----------|
| **Serif** (Times, Garamond, Playfair) | Traditional, trustworthy, authoritative, refined | Law, finance, editorial, luxury, education, heritage |
| **Sans-serif** (Helvetica, Inter, Montserrat) | Modern, clean, approachable, neutral | Tech, SaaS, startups, healthcare, corporate |
| **Slab serif** (Rockwell, Roboto Slab) | Strong, confident, bold, contemporary | Automotive, sports, construction, media |
| **Script/Handwritten** (Pacifico, Dancing Script) | Personal, elegant, creative, emotional | Fashion, beauty, food, wedding, boutique |
| **Display/Decorative** (Lobster, Righteous) | Unique, bold, attention-grabbing, playful | Entertainment, food, children, events |
| **Geometric** (Futura, Poppins, Century Gothic) | Progressive, forward-thinking, balanced | Tech, design, architecture, innovation |
| **Monospace** (JetBrains Mono, IBM Plex Mono) | Technical, precise, developer-oriented | Dev tools, coding platforms, cybersecurity |

### 8.5 Brand Personality to Design Mapping

Use this table to translate brand personality inputs into design parameters:

| Brand Personality | Colors | Shapes | Typography | Logo Type |
|------------------|--------|--------|------------|-----------|
| **Professional/Corporate** | Blue, gray, navy | Squares, clean lines | Sans-serif, serif | Wordmark, combination |
| **Playful/Fun** | Bright primaries, orange, yellow | Circles, curves, organic | Rounded sans, display | Mascot, combination |
| **Luxury/Premium** | Black, gold, deep purple | Thin lines, minimal shapes | Thin serif, elegant sans | Wordmark, emblem |
| **Eco/Natural** | Green, earth tones, teal | Organic curves, leaves | Humanist sans, rounded | Symbol, combination |
| **Bold/Disruptive** | Red, black, electric blue | Angular, triangles, sharp | Heavy sans, condensed | Abstract, wordmark |
| **Warm/Approachable** | Warm tones, peach, coral | Circles, rounded shapes | Rounded sans, friendly | Combination, mascot |
| **Technical/Precise** | Blue, white, cyan | Geometric, grid-based | Geometric sans, monospace | Abstract, lettermark |
| **Creative/Artistic** | Purple, magenta, mixed bold | Irregular, organic, asymmetric | Script, display, unique | Symbol, combination |
| **Heritage/Traditional** | Brown, navy, burgundy, gold | Shields, crests, classic shapes | Serif, old-style | Emblem, wordmark |
| **Minimalist/Modern** | Black, white, single accent | Simple geometry, negative space | Clean sans-serif | Wordmark, abstract |

---

## 9. Implementation Checklist for the AI Logo Generator

### 9.1 Input Collection
- [ ] Brand name (required)
- [ ] Industry / business type (required)
- [ ] Brand personality keywords (select 2-3 from predefined list)
- [ ] Target audience description
- [ ] Preferred logo type (or "auto-recommend" using decision tree in Section 3)
- [ ] Preferred colors (or "auto-suggest" using Section 8.1-8.2)
- [ ] Style preference (modern, classic, playful, bold, minimal, etc.)
- [ ] Any symbols or concepts to incorporate (optional)
- [ ] Any elements to avoid (optional)

### 9.2 Generation Pipeline
- [ ] Construct prompt using Layer 1-6 framework (Section 6.1)
- [ ] Generate 4-8 candidates per request
- [ ] Run through Technical Filter (binary pass/fail)
- [ ] Score each passing candidate (Section 7)
- [ ] Ensure diversity among top results
- [ ] Present top 3-4 results to user with scores

### 9.3 Output Deliverables
- [ ] Full color logo on white background
- [ ] Full color logo on dark background
- [ ] Monochrome (black) version
- [ ] Monochrome (white/reversed) version
- [ ] Icon-only variant (if combination mark)
- [ ] Wordmark-only variant (if combination mark)
- [ ] Multiple file formats (SVG, PNG at multiple sizes, PDF)
- [ ] Brand color codes (HEX, RGB, CMYK)
- [ ] Font identification or recommendation

---

## Sources

- [Logo Design Trends 2026 - ManyPixels](https://www.manypixels.co/blog/brand-design/logo-design-trends)
- [8 Logo Design Trends 2026 - Wix](https://www.wix.com/blog/logo-design-trends)
- [Top 10 Logo Design Trends 2026 - Digital Synopsis](https://digitalsynopsis.com/design/logo-design-trends-2026/)
- [Logo Design Trends 2026 - Creative Bloq](https://www.creativebloq.com/design/logos-icons/these-logo-design-trends-will-define-2026)
- [Top 7 Logo Design Trends 2026 - Onething Design](https://www.onething.design/post/top-7-logo-design-trends-in-2026)
- [7 Types of Logos - VistaPrint](https://www.vistaprint.com/hub/types-of-logos)
- [Types of Logos - Ebaq Design](https://www.ebaqdesign.com/blog/types-of-logos)
- [Types of Logos - Looka](https://looka.com/blog/different-types-of-logos/)
- [Logo Mistakes - Looka](https://looka.com/blog/logo-mistakes/)
- [Bad Logo Design - VistaPrint](https://www.vistaprint.com/hub/bad-logo-design)
- [Color Psychology - 99designs](https://99designs.com/logo-design/psychology-of-color)
- [Psychology of Logo Design - Wix](https://www.wix.com/blog/logo-psychology)
- [Logo Shape Psychology - Adobe](https://www.adobe.com/express/learn/blog/guide-to-logo-shapes)
- [Psychology of Shapes - Ebaq Design](https://www.ebaqdesign.com/blog/logo-shapes)
- [AI Prompts for Logo Design - Superside](https://www.superside.com/blog/ai-prompts-logo-design)
- [Art of Prompting for Logo AI - God of Prompt](https://www.godofprompt.ai/blog/the-art-of-prompting)
- [AI Logo Prompts - AND Academy](https://www.andacademy.com/resources/blog/graphic-design/ai-prompts-for-logo-design/)
