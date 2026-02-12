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
      scale: 2, // Improve quality
      useCORS: true,
      logging: false,
      backgroundColor: '#0f172a', // Match the slate-900 background
      onclone: (doc) => {
        const root = doc.getElementById(elementId) as HTMLElement | null;
        if (!root || !doc.defaultView) return;
        const win = doc.defaultView;
        
        // 1. Sanitize all <style> tags in the clone to prevent html2canvas crash on oklch/oklab/color-mix
        const styleTags = doc.getElementsByTagName('style');
        for (let i = 0; i < styleTags.length; i++) {
          const style = styleTags[i];
          if (style.textContent) {
            // Replace modern color functions with a safe fallback
            style.textContent = style.textContent
              .replace(/oklch\([^)]*\)/g, 'rgb(0,0,0)')
              .replace(/oklab\([^)]*\)/g, 'rgb(0,0,0)')
              .replace(/color-mix\([^)]*\)/g, 'rgb(0,0,0)');
          }
        }

        // Helper to normalize colors to RGB/Hex using a shared canvas context
        const canvas = doc.createElement('canvas');
        canvas.width = 1;
        canvas.height = 1;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        
        const formatColor = (color: string): string => {
           if (!color || color === 'transparent' || color === 'inherit') return color;
           if (color.startsWith('rgb') || color.startsWith('#') || color.startsWith('hsl')) return color;
           
           if (ctx) {
             try {
               ctx.fillStyle = color;
               return ctx.fillStyle; 
             } catch (e) {
               return color;
             }
           }
           return color;
        };

        // Fix Recharts sizing by applying fixed dimensions from the original element if possible
        const containers = root.querySelectorAll('.recharts-responsive-container');
        containers.forEach((container) => {
           const el = container as HTMLElement;
           if (!el.style.width && !el.style.height) {
              el.style.width = '100%';
              el.style.height = '100%';
           }
           el.style.display = 'block';
           el.style.overflow = 'visible';
        });

        const stack: HTMLElement[] = [root];
        while (stack.length) {
          const el = stack.pop()!;
          const cs = win.getComputedStyle(el);
          
          // Inline standard color properties
          const colorProps = [
            'color',
            'backgroundColor',
            'borderTopColor',
            'borderRightColor',
            'borderBottomColor',
            'borderLeftColor',
            'outlineColor',
            'textDecorationColor',
            'columnRuleColor',
            'fill',
            'stroke',
            'stopColor',
            'floodColor',
            'lightingColor'
          ] as const;
          
          for (const prop of colorProps) {
            const val = (cs as any)[prop];
            if (val) {
              (el.style as any)[prop] = formatColor(val);
            }
          }

          // Handle background-image (gradients) which may contain oklch/oklab
          const bgImage = cs.backgroundImage;
          if (bgImage && (bgImage.includes('okl') || bgImage.includes('lab(') || bgImage.includes('color-mix'))) {
             // If gradient contains unsupported colors, simplify it to a solid background
             // or just strip the gradient to avoid crash
             el.style.backgroundImage = 'none';
          }

          // Handle Shadows separately (sanitize oklab/oklch)
          const shadowProps = ['boxShadow', 'textShadow'] as const;
          for (const prop of shadowProps) {
             const val = (cs as any)[prop];
             if (val && (val.includes('okl') || val.includes('lab(') || val.includes('color-mix'))) {
                (el.style as any)[prop] = 'none';
             }
          }

          const children = el.children;
          for (let i = 0; i < children.length; i++) {
            const child = children[i];
            if (child.nodeType === 1) stack.push(child as HTMLElement);
          }
        }
      }
    });

    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4'
    });

    const imgWidth = 210; // A4 width in mm
    const pageHeight = 297; // A4 height in mm
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    let heightLeft = imgHeight;
    let position = 0;

    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
    heightLeft -= pageHeight;

    while (heightLeft >= 0) {
      position = heightLeft - imgHeight;
      pdf.addPage();
      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;
    }

    pdf.save(`${title.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.pdf`);
  } catch (error) {
    console.error('PDF Generation failed:', error);
  }
};
