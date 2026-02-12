import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

export const generatePDF = async (elementId: string, title: string = 'RefScore Analysis Report') => {
  const sourceElement = document.getElementById(elementId);
  if (!sourceElement) {
    console.error(`Element with id ${elementId} not found`);
    return;
  }

  // 1. Create a deep clone for manipulation to avoid touching the UI
  // We append it to body but hidden, so we can calculate layout
  const cloneContainer = document.createElement('div');
  cloneContainer.style.position = 'absolute';
  cloneContainer.style.top = '-10000px';
  cloneContainer.style.left = '0';
  cloneContainer.style.width = '1200px'; // Force desktop width for consistent rendering
  cloneContainer.style.zIndex = '-1000';
  // Ensure the background is dark as per the app theme
  cloneContainer.style.backgroundColor = '#0f172a'; 
  document.body.appendChild(cloneContainer);

  try {
    const clone = sourceElement.cloneNode(true) as HTMLElement;
    cloneContainer.appendChild(clone);

    // 2. Pre-process the clone for PDF layout
    
    // A. Fix Table: Remove scroll limits to show full content
    // Find any element with overflow-auto or max-h set
    const scrollables = clone.querySelectorAll('.overflow-x-auto, .overflow-y-auto, .max-h-\\[600px\\]');
    scrollables.forEach(el => {
      (el as HTMLElement).style.overflow = 'visible';
      (el as HTMLElement).style.maxHeight = 'none';
      (el as HTMLElement).style.height = 'auto';
    });

    // B. Fix Bottom Grid: Stack Table and Insights vertically
    // The table is usually in a col-span-2 div. We find it and force the parent to column layout.
    const tableSection = clone.querySelector('.lg\\:col-span-2');
    if (tableSection && tableSection.parentElement) {
      const parent = tableSection.parentElement;
      parent.style.display = 'flex';
      parent.style.flexDirection = 'column';
      parent.style.gap = '2rem';
      // Reset widths to full
      (tableSection as HTMLElement).style.width = '100%';
      // Find siblings (Insights) and reset their width too
      Array.from(parent.children).forEach(child => {
        (child as HTMLElement).style.width = '100%';
      });
    }

    // C. Fix Recharts: Lock dimensions to pixel values
    // In the offscreen clone, the browser has already calculated layout (since we appended to body).
    // We explicitly set the style.width/height to the computed pixel values.
    const charts = clone.querySelectorAll('.recharts-responsive-container');
    charts.forEach(chart => {
      const el = chart as HTMLElement;
      // Use the computed size from the source element if possible, or the clone's computed size
      // Since clone is 1200px wide, it should have good sizes.
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        el.style.width = `${rect.width}px`;
        el.style.height = `${rect.height}px`;
        el.style.display = 'block';
      } else {
        // Fallback if clone hasn't laid out yet (rare given we appended it)
        el.style.width = '100%';
        el.style.height = '400px'; 
      }
    });

    // 3. Initialize PDF
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4'
    });
    
    const pageWidth = 210;
    const pageHeight = 297;
    const margin = 10;
    const contentWidth = pageWidth - (margin * 2);
    
    let currentY = margin;

    // 4. Capture Sections
    // We look for elements marked with 'pdf-section'
    const sections = Array.from(clone.querySelectorAll('.pdf-section'));
    
    // If no sections found (fallback), just capture the whole thing
    const targets = sections.length > 0 ? sections : [clone];

    for (const section of targets) {
      // Skip hidden or empty sections
      if ((section as HTMLElement).style.display === 'none') continue;

      const canvas = await html2canvas(section as HTMLElement, {
        scale: 2, // Higher scale for better quality
        useCORS: true,
        logging: false,
        backgroundColor: '#0f172a', // Match app background
        onclone: (doc) => {
          // 5. Sanitize Colors inside html2canvas context
          // This prevents the "oklab" error by removing modern CSS from style tags
          const styleTags = doc.getElementsByTagName('style');
          for (let i = 0; i < styleTags.length; i++) {
            if (styleTags[i].textContent) {
              styleTags[i].textContent = styleTags[i].textContent!
                .replace(/oklch\([^)]*\)/g, 'rgb(0,0,0)')
                .replace(/oklab\([^)]*\)/g, 'rgb(0,0,0)')
                .replace(/color-mix\([^)]*\)/g, 'rgb(0,0,0)');
            }
          }

          // Also sanitize inline styles of the specific element tree
          const allElements = doc.querySelectorAll('*');
          allElements.forEach(el => {
             const htmlEl = el as HTMLElement;
             if (htmlEl.style) {
                // Check background images for gradients with oklab
                const bg = htmlEl.style.backgroundImage;
                if (bg && (bg.includes('okl') || bg.includes('lab('))) {
                   htmlEl.style.backgroundImage = 'none';
                }
                // Check box shadows
                const shadow = htmlEl.style.boxShadow;
                if (shadow && (shadow.includes('okl') || shadow.includes('lab('))) {
                   htmlEl.style.boxShadow = 'none';
                }
             }
          });
        }
      });

      const imgData = canvas.toDataURL('image/png');
      const imgHeight = (canvas.height * contentWidth) / canvas.width;

      // Smart Page Break Logic
      if (currentY + imgHeight > pageHeight - margin) {
        // If the section is huge (larger than a page), we have to add it anyway
        // But if it just doesn't fit the remaining space, add a new page
        if (currentY > margin) {
           pdf.addPage();
           currentY = margin;
        }
      }

      pdf.addImage(imgData, 'PNG', margin, currentY, contentWidth, imgHeight);
      currentY += imgHeight + 5; // 5mm spacing between sections
    }

    // Save PDF
    pdf.save(`${title.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.pdf`);

  } catch (error) {
    console.error('PDF Generation failed:', error);
    alert('PDF Generation failed. Please check console for details.');
  } finally {
    // Cleanup
    if (document.body.contains(cloneContainer)) {
      document.body.removeChild(cloneContainer);
    }
  }
};
