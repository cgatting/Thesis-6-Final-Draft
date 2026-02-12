import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

export const generatePDF = async (elementId: string, title: string = 'RefScore Analysis Report') => {
  const element = document.getElementById(elementId);
  if (!element) {
    console.error(`Element with id ${elementId} not found`);
    return;
  }

  try {
    const canvas = await html2canvas(element, {
      scale: 2,
      useCORS: true,
      logging: false,
      backgroundColor: '#0f172a',
      onclone: (doc) => {
        const root = doc.getElementById(elementId) as HTMLElement | null;
        if (!root || !doc.defaultView) return;
        const win = doc.defaultView;

        // 1. Setup color normalization canvas
        const colorCanvas = doc.createElement('canvas');
        colorCanvas.width = 1;
        colorCanvas.height = 1;
        const ctx = colorCanvas.getContext('2d', { medicalUsage: false } as any);

        const formatToRGB = (color: string): string => {
          if (!color || ['transparent', 'inherit', 'initial', 'unset'].includes(color)) return color;
          // If already a standard format, return as is
          if (color.startsWith('rgb') || color.startsWith('#') || color.startsWith('hsl')) return color;
          
          if (ctx) {
            try {
              // The browser's canvas engine will convert oklch/oklab to RGB automatically
              (ctx as CanvasRenderingContext2D).fillStyle = color;
              const normalized = (ctx as CanvasRenderingContext2D).fillStyle;
              // If the result is the same string and it's not a hex/rgb, it failed to parse
              if (normalized === color && !color.startsWith('#') && !color.startsWith('rgb')) {
                return 'rgb(0,0,0)'; // Fallback for truly unparseable modern functions
              }
              return normalized as string;
            } catch (e) {
              return 'rgb(0,0,0)';
            }
          }
          return color;
        };

        // 2. Fix Recharts Responsive Containers
        const containers = root.querySelectorAll('.recharts-responsive-container');
        containers.forEach((container) => {
          const el = container as HTMLElement;
          el.style.width = el.offsetWidth + 'px';
          el.style.height = el.offsetHeight + 'px';
          el.style.display = 'block';
          el.style.visibility = 'visible';
        });

        // 3. Recursively inline all styles and sanitize colors
        const stack: HTMLElement[] = [root];
        const colorProps = [
          'color', 'backgroundColor', 'borderTopColor', 'borderRightColor', 
          'borderBottomColor', 'borderLeftColor', 'outlineColor', 
          'fill', 'stroke', 'stopColor', 'floodColor'
        ] as const;

        while (stack.length) {
          const el = stack.pop()!;
          const cs = win.getComputedStyle(el);
          
          // Inline standard colors
          for (const prop of colorProps) {
            const val = (cs as any)[prop];
            if (val) {
              (el.style as any)[prop] = formatToRGB(val);
            }
          }

          // Strip problematic shadows and gradients
          const complexProps = ['boxShadow', 'textShadow', 'backgroundImage'] as const;
          for (const prop of complexProps) {
            const val = (cs as any)[prop];
            if (val && (val.includes('okl') || val.includes('lab(') || val.includes('color-mix'))) {
              (el.style as any)[prop] = 'none';
            }
          }

          // Ensure visibility
          el.style.opacity = '1';

          Array.from(el.children).forEach(child => {
            if (child instanceof HTMLElement) stack.push(child);
          });
        }

        // 4. CRITICAL: Remove all stylesheets to prevent html2canvas parser crash
        // Since we've inlined the styles we need on the elements themselves, 
        // we can safely remove the global CSS.
        const styles = doc.querySelectorAll('style, link[rel="stylesheet"]');
        styles.forEach(s => s.remove());
      }
    });

    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4'
    });

    const imgWidth = 210;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    let heightLeft = imgHeight;
    let position = 0;

    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
    heightLeft -= 297;

    while (heightLeft >= 0) {
      position = heightLeft - imgHeight;
      pdf.addPage();
      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
      heightLeft -= 297;
    }

    pdf.save(`${title.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.pdf`);
  } catch (error) {
    console.error('PDF Generation failed:', error);
  }
};