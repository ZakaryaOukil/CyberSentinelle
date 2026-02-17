import React, { useEffect, useRef } from 'react';

export default function HexGrid({ color = '#00F0FF', opacity = 0.04 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const resize = () => {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    };
    resize();

    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    const size = 30;
    const gap = 4;

    const drawHex = (cx, cy, s, alpha) => {
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 3) * i - Math.PI / 6;
        const x = cx + s * Math.cos(angle);
        const y = cy + s * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = `${color}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    };

    let time = 0;
    let animId;

    const animate = () => {
      ctx.clearRect(0, 0, w, h);
      time += 0.005;

      const rowH = size * Math.sqrt(3) + gap;
      const colW = size * 1.5 + gap;

      for (let row = -1; row < h / rowH + 1; row++) {
        for (let col = -1; col < w / colW + 1; col++) {
          const x = col * colW;
          const y = row * rowH + (col % 2 ? rowH / 2 : 0);
          const dist = Math.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2);
          const wave = Math.sin(dist * 0.01 - time * 2) * 0.5 + 0.5;
          const alpha = opacity * wave;
          if (alpha > 0.005) drawHex(x, y, size, alpha);
        }
      }

      animId = requestAnimationFrame(animate);
    };
    animate();
    return () => cancelAnimationFrame(animId);
  }, [color, opacity]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ zIndex: 0 }}
    />
  );
}
