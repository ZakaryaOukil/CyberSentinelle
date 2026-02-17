import React, { useEffect, useRef } from 'react';

export default function NetworkTopology3D() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    };
    resize();

    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    const cx = w / 2;
    const cy = h / 2;

    // 3D projection
    const project = (x, y, z, rot) => {
      const cosR = Math.cos(rot);
      const sinR = Math.sin(rot);
      const rx = x * cosR - z * sinR;
      const rz = x * sinR + z * cosR;
      const scale = 300 / (300 + rz);
      return { px: cx + rx * scale * 120, py: cy + y * scale * 80, scale, z: rz };
    };

    // Nodes
    const nodes = [
      { x: 0, y: -0.3, z: 0, type: 'server', label: 'SERVEUR IDS', color: '#00F0FF' },
      { x: -1.2, y: 0.2, z: 0.8, type: 'firewall', label: 'PARE-FEU', color: '#FAFF00' },
      { x: 1.2, y: 0.2, z: 0.8, type: 'firewall', label: 'ROUTEUR', color: '#FAFF00' },
      { x: -1.8, y: 0.8, z: -0.5, type: 'client', label: 'CLIENT', color: '#00FF41' },
      { x: -0.8, y: 0.9, z: -0.8, type: 'client', label: 'CLIENT', color: '#00FF41' },
      { x: 0.8, y: 0.9, z: -0.8, type: 'client', label: 'CLIENT', color: '#00FF41' },
      { x: 1.8, y: 0.8, z: -0.5, type: 'client', label: 'CLIENT', color: '#00FF41' },
      { x: -2.2, y: -0.5, z: 0, type: 'attacker', label: 'ATTAQUANT', color: '#FF003C' },
      { x: 2.2, y: -0.7, z: -0.3, type: 'attacker', label: 'BOTNET', color: '#FF003C' },
    ];

    const connections = [
      [0, 1, '#00F0FF'], [0, 2, '#00F0FF'],
      [1, 3, '#00FF41'], [1, 4, '#00FF41'],
      [2, 5, '#00FF41'], [2, 6, '#00FF41'],
      [7, 1, '#FF003C'], [8, 2, '#FF003C'],
    ];

    // Floating data packets
    const packets = connections.map((c) => ({
      from: c[0], to: c[1], color: c[2], progress: Math.random(), speed: 0.003 + Math.random() * 0.004
    }));

    // Background particles
    const bgParticles = Array.from({ length: 100 }, () => ({
      x: (Math.random() - 0.5) * 4,
      y: (Math.random() - 0.5) * 3,
      z: (Math.random() - 0.5) * 4,
      alpha: 0.1 + Math.random() * 0.3,
      size: 0.5 + Math.random() * 1.5
    }));

    let rotation = 0;
    let animId;

    const drawNode = (ctx, px, py, scale, type, color, label) => {
      const s = scale * 12;
      ctx.save();
      
      // Glow
      ctx.shadowColor = color;
      ctx.shadowBlur = 15 * scale;
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.fillStyle = color + '20';

      if (type === 'server') {
        ctx.beginPath();
        ctx.rect(px - s, py - s * 1.2, s * 2, s * 2.4);
        ctx.fill(); ctx.stroke();
        // Inner lines
        ctx.strokeStyle = color + '60';
        for (let i = 0; i < 3; i++) {
          ctx.beginPath();
          ctx.moveTo(px - s * 0.6, py - s * 0.6 + i * s * 0.6);
          ctx.lineTo(px + s * 0.6, py - s * 0.6 + i * s * 0.6);
          ctx.stroke();
        }
      } else if (type === 'firewall') {
        ctx.beginPath();
        ctx.moveTo(px, py - s * 1.2);
        ctx.lineTo(px + s, py);
        ctx.lineTo(px, py + s * 1.2);
        ctx.lineTo(px - s, py);
        ctx.closePath();
        ctx.fill(); ctx.stroke();
      } else if (type === 'attacker') {
        ctx.beginPath();
        ctx.moveTo(px, py - s * 1.2);
        ctx.lineTo(px + s, py + s * 0.8);
        ctx.lineTo(px - s, py + s * 0.8);
        ctx.closePath();
        ctx.fill(); ctx.stroke();
        // Danger cross
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px - s * 0.3, py - s * 0.1);
        ctx.lineTo(px + s * 0.3, py + s * 0.5);
        ctx.moveTo(px + s * 0.3, py - s * 0.1);
        ctx.lineTo(px - s * 0.3, py + s * 0.5);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(px, py, s, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
      }

      // Label
      ctx.shadowBlur = 0;
      ctx.fillStyle = color + '90';
      ctx.font = `${Math.max(8, 9 * scale)}px 'Share Tech Mono', monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(label, px, py + s * 1.8 + 8);
      
      ctx.restore();
    };

    const animate = () => {
      ctx.clearRect(0, 0, w, h);
      rotation += 0.003;

      // Background particles
      bgParticles.forEach((p) => {
        const proj = project(p.x, p.y, p.z, rotation);
        if (proj.z > -2) {
          ctx.beginPath();
          ctx.arc(proj.px, proj.py, p.size * proj.scale, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(0, 240, 255, ${p.alpha * proj.scale})`;
          ctx.fill();
        }
      });

      // Grid floor
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.05)';
      ctx.lineWidth = 0.5;
      for (let i = -5; i <= 5; i++) {
        const p1 = project(i * 0.5, 1.5, -2, rotation);
        const p2 = project(i * 0.5, 1.5, 2, rotation);
        ctx.beginPath(); ctx.moveTo(p1.px, p1.py); ctx.lineTo(p2.px, p2.py); ctx.stroke();
        const p3 = project(-2.5, 1.5, i * 0.4, rotation);
        const p4 = project(2.5, 1.5, i * 0.4, rotation);
        ctx.beginPath(); ctx.moveTo(p3.px, p3.py); ctx.lineTo(p4.px, p4.py); ctx.stroke();
      }

      // Project all nodes
      const projectedNodes = nodes.map((n, i) => {
        const bobY = n.y + Math.sin(Date.now() * 0.002 + i) * 0.05;
        return { ...n, ...project(n.x, bobY, n.z, rotation), idx: i };
      });

      // Sort by z for depth
      projectedNodes.sort((a, b) => a.z - b.z);

      // Draw connections
      connections.forEach(([from, to, color]) => {
        const a = { ...nodes[from], ...project(nodes[from].x, nodes[from].y, nodes[from].z, rotation) };
        const b = { ...nodes[to], ...project(nodes[to].x, nodes[to].y, nodes[to].z, rotation) };
        
        ctx.beginPath();
        ctx.moveTo(a.px, a.py);
        ctx.lineTo(b.px, b.py);
        ctx.strokeStyle = color + '30';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Dashed overlay
        ctx.setLineDash([4, 6]);
        ctx.strokeStyle = color + '15';
        ctx.stroke();
        ctx.setLineDash([]);
      });

      // Animate packets
      packets.forEach((p) => {
        p.progress = (p.progress + p.speed) % 1;
        const from = nodes[p.from];
        const to = nodes[p.to];
        const t = p.progress;
        const mx = from.x + (to.x - from.x) * t;
        const my = from.y + (to.y - from.y) * t - Math.sin(t * Math.PI) * 0.15;
        const mz = from.z + (to.z - from.z) * t;
        const proj = project(mx, my, mz, rotation);
        
        ctx.beginPath();
        ctx.arc(proj.px, proj.py, 3 * proj.scale, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      // Draw nodes (sorted by depth)
      projectedNodes.forEach((n) => {
        drawNode(ctx, n.px, n.py, n.scale, n.type, n.color, n.label);
      });

      animId = requestAnimationFrame(animate);
    };

    animate();
    return () => cancelAnimationFrame(animId);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full"
      style={{ background: 'transparent', minHeight: '400px' }}
    />
  );
}
