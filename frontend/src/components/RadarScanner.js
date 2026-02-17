import React, { useEffect, useRef } from 'react';

export default function RadarScanner({ isScanning = false, result = null }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.min(w, h) * 0.42;

    let angle = 0;
    let animId;
    const blips = [];

    // Generate random blips
    for (let i = 0; i < 15; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = 0.2 + Math.random() * 0.75;
      blips.push({
        angle: a, radius: r,
        isAttack: Math.random() > 0.6,
        pulse: Math.random() * Math.PI * 2,
        size: 2 + Math.random() * 3
      });
    }

    const isAttack = result?.prediction === 'Attack';
    const baseColor = isAttack ? '#FF003C' : isScanning ? '#00F0FF' : '#00FF41';

    const animate = () => {
      ctx.clearRect(0, 0, w, h);

      // Concentric rings
      for (let i = 1; i <= 4; i++) {
        const r = (maxR / 4) * i;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = `${baseColor}15`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Cross hairs
      ctx.strokeStyle = `${baseColor}10`;
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(cx - maxR, cy); ctx.lineTo(cx + maxR, cy);
      ctx.moveTo(cx, cy - maxR); ctx.lineTo(cx, cy + maxR);
      // Diagonal
      ctx.moveTo(cx - maxR * 0.7, cy - maxR * 0.7); ctx.lineTo(cx + maxR * 0.7, cy + maxR * 0.7);
      ctx.moveTo(cx + maxR * 0.7, cy - maxR * 0.7); ctx.lineTo(cx - maxR * 0.7, cy + maxR * 0.7);
      ctx.stroke();

      // Sweep beam
      const sweepAngle = angle;
      const sweepLen = 0.4;
      const gradient = ctx.createConicalGradient
        ? null
        : (() => {
            // Fallback: draw the sweep with arc
            ctx.save();
            ctx.translate(cx, cy);
            ctx.rotate(sweepAngle);

            const grad = ctx.createLinearGradient(0, 0, maxR, 0);
            grad.addColorStop(0, `${baseColor}30`);
            grad.addColorStop(1, `${baseColor}00`);

            for (let a = 0; a < sweepLen; a += 0.01) {
              ctx.save();
              ctx.rotate(-a);
              const alpha = (1 - a / sweepLen) * 0.3;
              ctx.beginPath();
              ctx.moveTo(0, 0);
              ctx.lineTo(maxR, 0);
              ctx.strokeStyle = `${baseColor}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`;
              ctx.lineWidth = 2;
              ctx.stroke();
              ctx.restore();
            }

            // Main sweep line
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(maxR, 0);
            ctx.strokeStyle = `${baseColor}80`;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.restore();
          })();

      // Blips
      blips.forEach((blip) => {
        blip.pulse += 0.05;
        const bx = cx + Math.cos(blip.angle) * blip.radius * maxR;
        const by = cy + Math.sin(blip.angle) * blip.radius * maxR;
        const pulseSize = blip.size + Math.sin(blip.pulse) * 1.5;

        // Blip trail (fade when sweep passes)
        const angleDiff = ((sweepAngle - blip.angle) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);
        const visibility = angleDiff < 1.5 ? (1 - angleDiff / 1.5) : 0.15;

        const blipColor = blip.isAttack ? '#FF003C' : '#00FF41';

        ctx.beginPath();
        ctx.arc(bx, by, pulseSize * visibility + 1, 0, Math.PI * 2);
        ctx.fillStyle = `${blipColor}${Math.round(visibility * 200).toString(16).padStart(2, '0')}`;
        ctx.fill();

        // Glow
        if (visibility > 0.5) {
          ctx.beginPath();
          ctx.arc(bx, by, pulseSize * 3, 0, Math.PI * 2);
          ctx.fillStyle = `${blipColor}10`;
          ctx.fill();
        }
      });

      // Center dot
      const centerPulse = 4 + Math.sin(Date.now() * 0.005) * 2;
      ctx.beginPath();
      ctx.arc(cx, cy, centerPulse, 0, Math.PI * 2);
      ctx.fillStyle = `${baseColor}60`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx, cy, 2, 0, Math.PI * 2);
      ctx.fillStyle = baseColor;
      ctx.fill();

      // Outer ring glow
      ctx.beginPath();
      ctx.arc(cx, cy, maxR, 0, Math.PI * 2);
      ctx.strokeStyle = `${baseColor}25`;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Labels
      ctx.font = "9px 'Share Tech Mono', monospace";
      ctx.fillStyle = `${baseColor}50`;
      ctx.textAlign = 'center';
      ctx.fillText('N', cx, cy - maxR - 8);
      ctx.fillText('S', cx, cy + maxR + 14);
      ctx.fillText('E', cx + maxR + 12, cy + 3);
      ctx.fillText('W', cx - maxR - 12, cy + 3);

      if (isScanning) angle += 0.02;
      else angle += 0.005;

      animId = requestAnimationFrame(animate);
    };

    animate();
    return () => cancelAnimationFrame(animId);
  }, [isScanning, result]);

  return (
    <canvas ref={canvasRef} className="w-full h-full" style={{ minHeight: '250px' }} />
  );
}
