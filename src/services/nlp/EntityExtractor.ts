export class EntityExtractor {
  /**
   * Extract potential entities from text using heuristic rules.
   * @param text Input text
   */
  public extract(text: string): string[] {
    const words = text.split(/\s+/);
    const entities = new Set<string>();
    
    words.forEach((word, idx) => {
      // Remove punctuation
      const cleanWord = word.replace(/[.,;!?()]/g, '');
      if (cleanWord.length < 3) return;
      
      // Check if capitalized
      // Improved: Check if it's NOT the start of a sentence or if it is, check if it's not a common word?
      // For now, keep the simple heuristic but filter common start words if possible.
      if (/^[A-Z]/.test(cleanWord)) {
        if (idx === 0) {
            // Skip first word unless it's all caps (e.g. NATO)
            if (!/^[A-Z]+$/.test(cleanWord)) return;
        }
        entities.add(cleanWord);
      }
    });

    return Array.from(entities);
  }
}
