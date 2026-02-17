import React, { useEffect, useRef } from 'react';

// CSS-based animated globe (no Three.js to avoid version conflicts)
export default function CyberGlobe({ showAlerts = false }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth * 2;
    const height = canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);
    
    const particles = [];
    const numParticles = 150;
    const centerX = width / 4;
    const centerY = height / 4;
    const radius = Math.min(width, height) / 5;
    
    // Create particles in sphere distribution
    for (let i = 0; i < numParticles; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);
      particles.push({
        theta,
        phi,
        speed: 0.002 + Math.random() * 0.003,
        size: 1 + Math.random() * 2,
        alpha: 0.3 + Math.random() * 0.7
      });
    }
    
    // Animation
    let animationId;
    let rotation = 0;
    
    const animate = () => {
      ctx.fillStyle = 'rgba(5, 5, 5, 0.1)';
      ctx.fillRect(0, 0, width / 2, height / 2);
      
      rotation += 0.005;
      
      // Draw particles
      particles.forEach(p => {
        const x = centerX + radius * Math.sin(p.phi) * Math.cos(p.theta + rotation);
        const y = centerY + radius * Math.sin(p.phi) * Math.sin(p.theta + rotation) * 0.3 + radius * Math.cos(p.phi);
        const z = Math.cos(p.theta + rotation);
        
        // Only draw if in front
        if (z > -0.3) {
          const alpha = p.alpha * (0.5 + z * 0.5);
          ctx.beginPath();
          ctx.arc(x, y, p.size * (0.5 + z * 0.5), 0, Math.PI * 2);
          ctx.fillStyle = `rgba(0, 240, 255, ${alpha})`;
          ctx.fill();
        }
      });
      
      // Draw rings
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.ellipse(centerX, centerY, radius * 1.2, radius * 0.4, 0, 0, Math.PI * 2);
      ctx.stroke();
      
      ctx.strokeStyle = 'rgba(189, 0, 255, 0.15)';
      ctx.beginPath();
      ctx.ellipse(centerX, centerY, radius * 1.4, radius * 0.5, 0.2, 0, Math.PI * 2);
      ctx.stroke();
      
      // Draw connection lines
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.05)';
      ctx.lineWidth = 0.5;
      for (let i = 0; i < particles.length - 1; i += 3) {
        const p1 = particles[i];
        const p2 = particles[i + 1];
        const x1 = centerX + radius * Math.sin(p1.phi) * Math.cos(p1.theta + rotation);
        const y1 = centerY + radius * Math.sin(p1.phi) * Math.sin(p1.theta + rotation) * 0.3 + radius * Math.cos(p1.phi);
        const x2 = centerX + radius * Math.sin(p2.phi) * Math.cos(p2.theta + rotation);
        const y2 = centerY + radius * Math.sin(p2.phi) * Math.sin(p2.theta + rotation) * 0.3 + radius * Math.cos(p2.phi);
        
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
      
      // Alert pulses if enabled
      if (showAlerts) {
        const pulseSize = 5 + Math.sin(Date.now() * 0.005) * 3;
        const pulseAlpha = 0.5 + Math.sin(Date.now() * 0.005) * 0.3;
        
        // Random alert positions
        [[0.3, 0.4], [0.7, 0.6], [0.5, 0.3], [0.4, 0.7]].forEach(([px, py], i) => {
          ctx.beginPath();
          ctx.arc(width / 4 * px + centerX * 0.5, height / 4 * py, pulseSize, 0, Math.PI * 2);
          ctx.fillStyle = i % 2 === 0 ? `rgba(255, 0, 60, ${pulseAlpha})` : `rgba(250, 255, 0, ${pulseAlpha})`;
          ctx.fill();
        });
      }
      
      animationId = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => cancelAnimationFrame(animationId);
  }, [showAlerts]);
  
  return (
    <canvas 
      ref={canvasRef} 
      className="w-full h-full"
      style={{ background: 'transparent' }}
    />
  );
}

// Mini version for cards
export function MiniGlobe() {
  return <CyberGlobe showAlerts={false} />;
}
