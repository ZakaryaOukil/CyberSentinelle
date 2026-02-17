import React, { useEffect, useRef } from 'react';

export default function ThreatVisualization3D() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width;
    const h = rect.height;
    const cx = w / 2;
    const cy = h / 2;

    // Shield icosahedron
    const phi = (1 + Math.sqrt(5)) / 2;
    const verts = [
      [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
      [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
      [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ];
    const len = Math.sqrt(1 + phi * phi);
    const shieldR = 50;
    const shieldVerts = verts.map(([x, y, z]) => ({
      x: (x / len) * shieldR, y: (y / len) * shieldR, z: (z / len) * shieldR
    }));
    const edges = [
      [0,1],[0,5],[0,7],[0,10],[0,11],[1,5],[1,7],[1,8],[1,9],
      [2,3],[2,4],[2,6],[2,10],[2,11],[3,4],[3,6],[3,8],[3,9],
      [4,5],[4,9],[4,11],[5,9],[5,11],[6,7],[6,8],[6,10],[7,8],[7,10],[8,9],[10,11]
    ];

    // Faces for semi-transparent fill
    const faces = [
      [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
      [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
      [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
      [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ];

    // Attack particles
    const attacks = Array.from({ length: 200 }, () => {
      const t = Math.random() * Math.PI * 2;
      const p = Math.acos(Math.random() * 2 - 1);
      const r = 100 + Math.random() * 100;
      return { theta: t, phi: p, r, speed: 0.002 + Math.random() * 0.006, size: 1 + Math.random() * 2.5 };
    });

    // Normal traffic
    const normals = Array.from({ length: 120 }, (_, i) => ({
      angle: (i / 120) * Math.PI * 2,
      r: 55 + Math.random() * 25,
      yOff: (Math.random() - 0.5) * 40,
      speed: 0.005 + Math.random() * 0.008,
      size: 1 + Math.random() * 1.5
    }));

    // Orbiting threat rings
    const rings = [
      { r: 90, tiltX: 1.57, tiltZ: 0, speed: 0.004, color: '#FF003C', w: 1.5, dash: [6, 4] },
      { r: 110, tiltX: 0.8, tiltZ: 0.5, speed: -0.003, color: '#BD00FF', w: 1, dash: [3, 6] },
      { r: 130, tiltX: 0.4, tiltZ: 1, speed: 0.002, color: '#FAFF00', w: 0.8, dash: [] },
      { r: 70, tiltX: 1.2, tiltZ: 0.3, speed: 0.006, color: '#00FF41', w: 1, dash: [2, 3] },
    ];

    // Data streams converging toward shield
    const streams = Array.from({ length: 30 }, () => ({
      angle: Math.random() * Math.PI * 2,
      startR: 150 + Math.random() * 50,
      progress: Math.random(),
      speed: 0.003 + Math.random() * 0.005,
      isAttack: Math.random() > 0.5
    }));

    const project3D = (x, y, z, rotY, rotX = 0.35) => {
      const cy2 = Math.cos(rotY), sy = Math.sin(rotY);
      const cx2 = Math.cos(rotX), sx = Math.sin(rotX);
      let rx = x * cy2 - z * sy;
      let rz = x * sy + z * cy2;
      let ry = y * cx2 - rz * sx;
      rz = y * sx + rz * cx2;
      const d = 400;
      const scale = d / (d + rz);
      return { px: cx + rx * scale, py: cy + ry * scale, scale, z: rz };
    };

    let time = 0;
    let animId;

    const animate = () => {
      time += 0.016;
      ctx.clearRect(0, 0, w, h);

      const rotY = time * 0.25;
      const rotX = 0.3 + Math.sin(time * 0.12) * 0.08;

      // Shield faces (semi-transparent)
      faces.forEach(([a, b, c]) => {
        const pa = project3D(shieldVerts[a].x, shieldVerts[a].y, shieldVerts[a].z, rotY, rotX);
        const pb = project3D(shieldVerts[b].x, shieldVerts[b].y, shieldVerts[b].z, rotY, rotX);
        const pc = project3D(shieldVerts[c].x, shieldVerts[c].y, shieldVerts[c].z, rotY, rotX);
        // Only draw front-facing
        const cross = (pb.px - pa.px) * (pc.py - pa.py) - (pb.py - pa.py) * (pc.px - pa.px);
        if (cross > 0) {
          ctx.beginPath();
          ctx.moveTo(pa.px, pa.py);
          ctx.lineTo(pb.px, pb.py);
          ctx.lineTo(pc.px, pc.py);
          ctx.closePath();
          ctx.fillStyle = 'rgba(0, 240, 255, 0.03)';
          ctx.fill();
        }
      });

      // Shield wireframe edges
      edges.forEach(([a, b]) => {
        const pa = project3D(shieldVerts[a].x, shieldVerts[a].y, shieldVerts[a].z, rotY, rotX);
        const pb = project3D(shieldVerts[b].x, shieldVerts[b].y, shieldVerts[b].z, rotY, rotX);
        ctx.beginPath();
        ctx.moveTo(pa.px, pa.py);
        ctx.lineTo(pb.px, pb.py);
        ctx.strokeStyle = `rgba(0, 240, 255, ${0.2 * Math.max(pa.scale, pb.scale)})`;
        ctx.lineWidth = 0.8;
        ctx.stroke();
      });

      // Shield vertex glow
      shieldVerts.forEach(v => {
        const p = project3D(v.x, v.y, v.z, rotY, rotX);
        if (p.z > -30) {
          ctx.beginPath();
          ctx.arc(p.px, p.py, 2.5 * p.scale, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(0, 240, 255, ${0.6 * p.scale})`;
          ctx.fill();
        }
      });

      // Central core
      const pulseR = 18 + Math.sin(time * 3) * 4;
      const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, pulseR * 3);
      coreGrad.addColorStop(0, 'rgba(0, 240, 255, 0.4)');
      coreGrad.addColorStop(0.3, 'rgba(0, 240, 255, 0.1)');
      coreGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = coreGrad;
      ctx.fillRect(cx - pulseR * 3, cy - pulseR * 3, pulseR * 6, pulseR * 6);

      ctx.beginPath();
      ctx.arc(cx, cy, pulseR, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 240, 255, ${0.25 + Math.sin(time * 3) * 0.1})`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#00F0FF';
      ctx.fill();

      // Orbiting rings
      rings.forEach(ring => {
        ctx.setLineDash(ring.dash);
        ctx.strokeStyle = `${ring.color}35`;
        ctx.lineWidth = ring.w;
        ctx.beginPath();
        for (let i = 0; i <= 80; i++) {
          const a = (i / 80) * Math.PI * 2;
          const px3 = Math.cos(a) * ring.r;
          const py3 = Math.sin(a) * ring.r * Math.sin(ring.tiltX);
          const pz3 = Math.sin(a) * ring.r * Math.cos(ring.tiltX);
          const p = project3D(px3, py3, pz3, rotY + time * ring.speed, rotX);
          if (i === 0) ctx.moveTo(p.px, p.py); else ctx.lineTo(p.px, p.py);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      });

      // Normal traffic (green orbiting)
      normals.forEach(p => {
        const a = p.angle + time * p.speed;
        const p3 = project3D(Math.cos(a) * p.r, p.yOff + Math.sin(time + p.angle) * 5, Math.sin(a) * p.r, rotY, rotX);
        if (p3.z > -50) {
          ctx.beginPath();
          ctx.arc(p3.px, p3.py, p.size * p3.scale, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(0, 255, 65, ${0.5 * p3.scale})`;
          ctx.fill();
        }
      });

      // Attack particles (red, outer)
      attacks.forEach(p => {
        const a = p.theta + time * p.speed;
        const p3 = project3D(
          p.r * Math.sin(p.phi) * Math.cos(a),
          p.r * Math.sin(p.phi) * Math.sin(a),
          p.r * Math.cos(p.phi), rotY * 0.5, rotX
        );
        if (p3.z > -60) {
          ctx.beginPath();
          ctx.arc(p3.px, p3.py, p.size * p3.scale, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 0, 60, ${0.4 * p3.scale})`;
          ctx.fill();
        }
      });

      // Data streams converging
      streams.forEach(s => {
        s.progress = (s.progress + s.speed) % 1;
        const t = s.progress;
        const r = s.startR * (1 - t);
        const sx = Math.cos(s.angle) * r;
        const sy = Math.sin(s.angle * 0.5) * r * 0.3;
        const sz = Math.sin(s.angle) * r;
        const p = project3D(sx, sy, sz, rotY, rotX);
        const color = s.isAttack ? '#FF003C' : '#00FF41';
        const alpha = t * 0.6;

        ctx.beginPath();
        ctx.arc(p.px, p.py, 1.5 * p.scale, 0, Math.PI * 2);
        ctx.fillStyle = `${color}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`;
        ctx.fill();
      });

      // Scan line
      const scanY = cy + Math.sin(time * 1.2) * 100;
      const scanGrad = ctx.createLinearGradient(cx - 160, scanY, cx + 160, scanY);
      scanGrad.addColorStop(0, 'transparent');
      scanGrad.addColorStop(0.5, 'rgba(0, 240, 255, 0.06)');
      scanGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = scanGrad;
      ctx.fillRect(cx - 160, scanY - 1, 320, 2);

      // HUD corners
      ctx.strokeStyle = '#00F0FF15';
      ctx.lineWidth = 1;
      // Top-left
      ctx.beginPath(); ctx.moveTo(10, 30); ctx.lineTo(10, 10); ctx.lineTo(30, 10); ctx.stroke();
      // Top-right
      ctx.beginPath(); ctx.moveTo(w - 30, 10); ctx.lineTo(w - 10, 10); ctx.lineTo(w - 10, 30); ctx.stroke();
      // Bottom-left
      ctx.beginPath(); ctx.moveTo(10, h - 30); ctx.lineTo(10, h - 10); ctx.lineTo(30, h - 10); ctx.stroke();
      // Bottom-right
      ctx.beginPath(); ctx.moveTo(w - 30, h - 10); ctx.lineTo(w - 10, h - 10); ctx.lineTo(w - 10, h - 30); ctx.stroke();

      // HUD text
      ctx.fillStyle = '#00F0FF25';
      ctx.font = "9px 'Share Tech Mono', monospace";
      ctx.textAlign = 'left';
      ctx.fillText(`SHIELD: ACTIVE`, 15, h - 18);
      ctx.textAlign = 'right';
      ctx.fillText(`THREATS: ${attacks.length}`, w - 15, h - 18);

      animId = requestAnimationFrame(animate);
    };

    animate();
    return () => cancelAnimationFrame(animId);
  }, []);

  return <canvas ref={canvasRef} className="w-full h-full" style={{ minHeight: '500px' }} />;
}
