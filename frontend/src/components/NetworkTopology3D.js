import React, { useEffect, useRef } from 'react';

export default function NetworkTopology3D() {
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
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();

    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    const cx = w / 2;
    const cy = h / 2;

    const project = (x, y, z, rot, tilt = 0.35) => {
      const cr = Math.cos(rot), sr = Math.sin(rot);
      const ct = Math.cos(tilt), st = Math.sin(tilt);
      let rx = x * cr - z * sr;
      let rz = x * sr + z * cr;
      let ry = y * ct - rz * st;
      rz = y * st + rz * ct;
      const d = 350;
      const scale = d / (d + rz);
      return { px: cx + rx * scale * 100, py: cy + ry * scale * 70, scale, z: rz };
    };

    // Network nodes
    const nodes = [
      // Central server cluster
      { x: 0, y: -0.2, z: 0, type: 'server', label: 'SERVEUR IDS', size: 1.2 },
      { x: 0.3, y: -0.35, z: 0.2, type: 'server', label: 'DB', size: 0.8 },
      { x: -0.3, y: -0.1, z: 0.15, type: 'server', label: 'LOG', size: 0.7 },
      // Firewalls
      { x: -1.8, y: 0.1, z: 1, type: 'firewall', label: 'PARE-FEU A', size: 1 },
      { x: 1.8, y: 0.1, z: 1, type: 'firewall', label: 'PARE-FEU B', size: 1 },
      { x: 0, y: 0.3, z: -1.5, type: 'firewall', label: 'WAF', size: 0.9 },
      // Clients
      { x: -2.5, y: 0.7, z: -0.5, type: 'client', label: 'POSTE 01', size: 0.7 },
      { x: -1.5, y: 0.8, z: -1.2, type: 'client', label: 'POSTE 02', size: 0.7 },
      { x: 0, y: 0.9, z: -2.2, type: 'client', label: 'POSTE 03', size: 0.7 },
      { x: 1.5, y: 0.8, z: -1.2, type: 'client', label: 'POSTE 04', size: 0.7 },
      { x: 2.5, y: 0.7, z: -0.5, type: 'client', label: 'POSTE 05', size: 0.7 },
      { x: -1, y: 0.85, z: -1.8, type: 'client', label: 'IoT 01', size: 0.5 },
      { x: 1, y: 0.85, z: -1.8, type: 'client', label: 'IoT 02', size: 0.5 },
      // Attackers
      { x: -3, y: -0.5, z: 0.5, type: 'attacker', label: 'ATTAQUANT', size: 0.9 },
      { x: 3.2, y: -0.3, z: 0, type: 'attacker', label: 'BOTNET C2', size: 0.9 },
      { x: -2.5, y: -0.8, z: -0.8, type: 'attacker', label: 'SCANNER', size: 0.7 },
      { x: 2.8, y: -0.7, z: -1, type: 'attacker', label: 'DDoS BOT', size: 0.7 },
    ];

    const connections = [
      // Server cluster internal
      [0, 1, '#00F0FF', 0.4], [0, 2, '#00F0FF', 0.4],
      // Server to firewalls
      [0, 3, '#00F0FF', 0.6], [0, 4, '#00F0FF', 0.6], [0, 5, '#00F0FF', 0.5],
      // Firewalls to clients
      [3, 6, '#00FF41', 0.5], [3, 7, '#00FF41', 0.5],
      [5, 8, '#00FF41', 0.5], [5, 11, '#00FF41', 0.4],
      [4, 9, '#00FF41', 0.5], [4, 10, '#00FF41', 0.5],
      [5, 12, '#00FF41', 0.4],
      // Attacks
      [13, 3, '#FF003C', 0.8], [14, 4, '#FF003C', 0.8],
      [15, 3, '#BD00FF', 0.6], [16, 4, '#BD00FF', 0.6],
    ];

    // Data packets
    const packets = connections.map((c) => ({
      from: c[0], to: c[1], color: c[2],
      progress: Math.random(), speed: 0.002 + Math.random() * 0.004,
      size: c[2] === '#FF003C' ? 3 : 2
    }));
    // Extra packets for attacks
    for (let i = 0; i < 8; i++) {
      const atkConns = connections.filter(c => c[2] === '#FF003C' || c[2] === '#BD00FF');
      const c = atkConns[i % atkConns.length];
      packets.push({
        from: c[0], to: c[1], color: c[2],
        progress: Math.random(), speed: 0.003 + Math.random() * 0.006, size: 2.5
      });
    }

    // Background particles
    const bgParticles = Array.from({ length: 150 }, () => ({
      x: (Math.random() - 0.5) * 8, y: (Math.random() - 0.5) * 5, z: (Math.random() - 0.5) * 8,
      alpha: 0.05 + Math.random() * 0.2, size: 0.5 + Math.random() * 1.5
    }));

    // Hex grid on floor
    const hexSize = 0.3;
    const hexGrid = [];
    for (let gx = -4; gx <= 4; gx++) {
      for (let gz = -3; gz <= 3; gz++) {
        const ox = gx * hexSize * 1.8 + (gz % 2 ? hexSize * 0.9 : 0);
        hexGrid.push({ x: ox, z: gz * hexSize * 1.6 });
      }
    }

    let rotation = 0;
    let animId;

    const drawShape = (ctx, px, py, scale, type, color, size, label) => {
      const s = scale * 10 * size;
      ctx.save();
      ctx.shadowColor = color;
      ctx.shadowBlur = 12 * scale;

      // Outer glow ring
      ctx.beginPath();
      ctx.arc(px, py, s * 1.8, 0, Math.PI * 2);
      ctx.fillStyle = `${color}08`;
      ctx.fill();

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.fillStyle = `${color}18`;

      if (type === 'server') {
        // Hexagon
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
          const a = (Math.PI / 3) * i - Math.PI / 6;
          const px2 = px + s * 1.2 * Math.cos(a);
          const py2 = py + s * 1.2 * Math.sin(a);
          if (i === 0) ctx.moveTo(px2, py2); else ctx.lineTo(px2, py2);
        }
        ctx.closePath();
        ctx.fill(); ctx.stroke();
        // Inner core
        ctx.beginPath();
        ctx.arc(px, py, s * 0.4, 0, Math.PI * 2);
        ctx.fillStyle = `${color}60`;
        ctx.fill();
      } else if (type === 'firewall') {
        // Diamond with inner lines
        ctx.beginPath();
        ctx.moveTo(px, py - s * 1.3);
        ctx.lineTo(px + s, py);
        ctx.lineTo(px, py + s * 1.3);
        ctx.lineTo(px - s, py);
        ctx.closePath();
        ctx.fill(); ctx.stroke();
        // Shield icon
        ctx.beginPath();
        ctx.moveTo(px, py - s * 0.5);
        ctx.lineTo(px + s * 0.4, py - s * 0.15);
        ctx.lineTo(px + s * 0.4, py + s * 0.3);
        ctx.lineTo(px, py + s * 0.5);
        ctx.lineTo(px - s * 0.4, py + s * 0.3);
        ctx.lineTo(px - s * 0.4, py - s * 0.15);
        ctx.closePath();
        ctx.strokeStyle = `${color}80`;
        ctx.lineWidth = 1;
        ctx.stroke();
      } else if (type === 'attacker') {
        // Skull-like shape
        ctx.beginPath();
        ctx.moveTo(px, py - s * 1.4);
        ctx.lineTo(px + s * 1.1, py + s * 0.5);
        ctx.lineTo(px, py + s * 0.1);
        ctx.lineTo(px - s * 1.1, py + s * 0.5);
        ctx.closePath();
        ctx.fill(); ctx.stroke();
        // Warning X
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px - s * 0.3, py - s * 0.3);
        ctx.lineTo(px + s * 0.3, py + s * 0.1);
        ctx.moveTo(px + s * 0.3, py - s * 0.3);
        ctx.lineTo(px - s * 0.3, py + s * 0.1);
        ctx.stroke();
      } else {
        // Client - circle with ring
        ctx.beginPath();
        ctx.arc(px, py, s * 0.8, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
        ctx.beginPath();
        ctx.arc(px, py, s * 0.4, 0, Math.PI * 2);
        ctx.fillStyle = `${color}40`;
        ctx.fill();
      }

      // Label
      ctx.shadowBlur = 0;
      ctx.fillStyle = `${color}80`;
      ctx.font = `${Math.max(7, 8 * scale)}px 'Share Tech Mono', monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(label, px, py + s * 2 + 6);
      ctx.restore();
    };

    const animate = () => {
      ctx.clearRect(0, 0, w, h);
      rotation += 0.002;

      // Hex grid floor
      ctx.strokeStyle = '#00F0FF08';
      ctx.lineWidth = 0.5;
      hexGrid.forEach(({ x, z }) => {
        const p = project(x, 1.8, z, rotation);
        if (p.z > -3 && p.scale > 0.3) {
          ctx.beginPath();
          for (let i = 0; i < 6; i++) {
            const a = (Math.PI / 3) * i;
            const hx = p.px + hexSize * 18 * p.scale * Math.cos(a);
            const hy = p.py + hexSize * 18 * p.scale * Math.sin(a);
            if (i === 0) ctx.moveTo(hx, hy); else ctx.lineTo(hx, hy);
          }
          ctx.closePath();
          ctx.stroke();
        }
      });

      // Background particles
      bgParticles.forEach(p => {
        const proj = project(p.x, p.y, p.z, rotation);
        if (proj.z > -3) {
          ctx.beginPath();
          ctx.arc(proj.px, proj.py, p.size * proj.scale, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(0, 240, 255, ${p.alpha * proj.scale})`;
          ctx.fill();
        }
      });

      // Project and sort nodes
      const projected = nodes.map((n, i) => {
        const bobY = n.y + Math.sin(Date.now() * 0.001 + i * 0.7) * 0.04;
        const p = project(n.x, bobY, n.z, rotation);
        return { ...n, ...p, idx: i };
      });

      // Draw connections
      connections.forEach(([from, to, color, width]) => {
        const a = projected[from], b = projected[to];
        
        // Glow effect
        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.moveTo(a.px, a.py);
        // Curved connection
        const midX = (a.px + b.px) / 2;
        const midY = (a.py + b.py) / 2 - 10;
        ctx.quadraticCurveTo(midX, midY, b.px, b.py);
        ctx.strokeStyle = `${color}18`;
        ctx.lineWidth = width || 1;
        ctx.stroke();
        ctx.restore();
      });

      // Animate packets
      packets.forEach(p => {
        p.progress = (p.progress + p.speed) % 1;
        const from = projected[p.from], to = projected[p.to];
        const t = p.progress;
        const px2 = from.px + (to.px - from.px) * t;
        const py2 = from.py + (to.py - from.py) * t - Math.sin(t * Math.PI) * 12;

        // Trail
        for (let trail = 0; trail < 3; trail++) {
          const tt = Math.max(0, t - trail * 0.03);
          const tpx = from.px + (to.px - from.px) * tt;
          const tpy = from.py + (to.py - from.py) * tt - Math.sin(tt * Math.PI) * 12;
          ctx.beginPath();
          ctx.arc(tpx, tpy, (p.size - trail * 0.5) * 0.7, 0, Math.PI * 2);
          ctx.fillStyle = `${p.color}${Math.round((1 - trail / 3) * 80).toString(16).padStart(2, '0')}`;
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(px2, py2, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      // Draw nodes sorted by depth
      const sorted = [...projected].sort((a, b) => a.z - b.z);
      sorted.forEach(n => {
        const colors = { server: '#00F0FF', client: '#00FF41', attacker: '#FF003C', firewall: '#FAFF00' };
        drawShape(ctx, n.px, n.py, n.scale, n.type, colors[n.type], n.size, n.label);
      });

      // HUD overlay text
      ctx.fillStyle = '#00F0FF20';
      ctx.font = "9px 'Share Tech Mono', monospace";
      ctx.textAlign = 'left';
      ctx.fillText(`NODES: ${nodes.length}`, 12, 20);
      ctx.fillText(`CONNECTIONS: ${connections.length}`, 12, 34);
      ctx.fillText(`THREATS: ${nodes.filter(n => n.type === 'attacker').length}`, 12, 48);

      animId = requestAnimationFrame(animate);
    };

    animate();
    return () => cancelAnimationFrame(animId);
  }, []);

  return <canvas ref={canvasRef} className="w-full h-full" style={{ minHeight: '500px' }} />;
}
