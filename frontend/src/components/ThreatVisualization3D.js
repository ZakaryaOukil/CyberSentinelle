import React, { useEffect, useRef } from 'react';

export default function ThreatVisualization3D() {
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

    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    const cx = w / 2;
    const cy = h / 2;

    // Attack particles (red, outer orbit)
    const attackParticles = Array.from({ length: 120 }, () => {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      const r = 120 + Math.random() * 80;
      return { theta, phi, r, speed: 0.003 + Math.random() * 0.005, size: 1.5 + Math.random() * 2 };
    });

    // Normal traffic (green, inner ring)
    const normalParticles = Array.from({ length: 80 }, (_, i) => ({
      angle: (i / 80) * Math.PI * 2,
      r: 60 + Math.random() * 20,
      speed: 0.008 + Math.random() * 0.006,
      yOff: (Math.random() - 0.5) * 30,
      size: 1 + Math.random() * 1.5,
    }));

    // Shield geometry (icosahedron points)
    const shieldRadius = 45;
    const shieldPoints = [];
    const goldenRatio = (1 + Math.sqrt(5)) / 2;
    const vertices = [
      [-1, goldenRatio, 0], [1, goldenRatio, 0], [-1, -goldenRatio, 0], [1, -goldenRatio, 0],
      [0, -1, goldenRatio], [0, 1, goldenRatio], [0, -1, -goldenRatio], [0, 1, -goldenRatio],
      [goldenRatio, 0, -1], [goldenRatio, 0, 1], [-goldenRatio, 0, -1], [-goldenRatio, 0, 1],
    ];
    const len = Math.sqrt(1 + goldenRatio * goldenRatio);
    vertices.forEach(([x, y, z]) => {
      shieldPoints.push({ x: (x / len) * shieldRadius, y: (y / len) * shieldRadius, z: (z / len) * shieldRadius });
    });

    const edges = [
      [0,1],[0,5],[0,7],[0,10],[0,11],[1,5],[1,7],[1,8],[1,9],
      [2,3],[2,4],[2,6],[2,10],[2,11],[3,4],[3,6],[3,8],[3,9],
      [4,5],[4,9],[4,11],[5,9],[5,11],[6,7],[6,8],[6,10],[7,8],[7,10],
      [8,9],[10,11]
    ];

    // Orbiting rings
    const rings = [
      { r: 100, tilt: 0.5, speed: 0.004, color: '#FF003C', width: 1.5 },
      { r: 115, tilt: 0.8, speed: -0.003, color: '#BD00FF', width: 1 },
      { r: 130, tilt: 0.3, speed: 0.002, color: '#FAFF00', width: 0.8 },
    ];

    let time = 0;
    let animId;

    const project3D = (x, y, z, rotY, rotX = 0.3) => {
      const cy2 = Math.cos(rotY), sy = Math.sin(rotY);
      const cx2 = Math.cos(rotX), sx = Math.sin(rotX);
      let rx = x * cy2 - z * sy;
      let rz = x * sy + z * cy2;
      let ry = y * cx2 - rz * sx;
      rz = y * sx + rz * cx2;
      const scale = 400 / (400 + rz);
      return { px: cx + rx * scale, py: cy + ry * scale, scale, z: rz };
    };

    const animate = () => {
      time += 0.016;
      ctx.clearRect(0, 0, w, h);

      const rotY = time * 0.3;
      const rotX = 0.3 + Math.sin(time * 0.15) * 0.1;

      // Shield wireframe
      ctx.strokeStyle = '#00F0FF30';
      ctx.lineWidth = 0.5;
      edges.forEach(([a, b]) => {
        const pa = project3D(shieldPoints[a].x, shieldPoints[a].y, shieldPoints[a].z, rotY, rotX);
        const pb = project3D(shieldPoints[b].x, shieldPoints[b].y, shieldPoints[b].z, rotY, rotX);
        ctx.beginPath();
        ctx.moveTo(pa.px, pa.py);
        ctx.lineTo(pb.px, pb.py);
        ctx.stroke();
      });

      // Shield vertices glow
      shieldPoints.forEach((p) => {
        const proj = project3D(p.x, p.y, p.z, rotY, rotX);
        if (proj.z > -30) {
          ctx.beginPath();
          ctx.arc(proj.px, proj.py, 2 * proj.scale, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(0, 240, 255, ${0.5 * proj.scale})`;
          ctx.fill();
        }
      });

      // Central core glow
      const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, 50);
      gradient.addColorStop(0, 'rgba(0, 240, 255, 0.3)');
      gradient.addColorStop(0.5, 'rgba(0, 240, 255, 0.08)');
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.fillRect(cx - 50, cy - 50, 100, 100);

      // Pulsing core
      const pulseR = 15 + Math.sin(time * 3) * 3;
      ctx.beginPath();
      ctx.arc(cx, cy, pulseR, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 240, 255, ${0.4 + Math.sin(time * 3) * 0.15})`;
      ctx.fill();

      // Orbiting rings
      rings.forEach((ring) => {
        ctx.strokeStyle = ring.color + '40';
        ctx.lineWidth = ring.width;
        ctx.beginPath();
        const steps = 60;
        for (let i = 0; i <= steps; i++) {
          const a = (i / steps) * Math.PI * 2;
          const px3d = Math.cos(a) * ring.r;
          const py3d = Math.sin(a) * ring.r * ring.tilt;
          const pz3d = Math.sin(a) * ring.r * (1 - ring.tilt);
          const proj = project3D(px3d, py3d, pz3d, rotY + time * ring.speed, rotX);
          if (i === 0) ctx.moveTo(proj.px, proj.py);
          else ctx.lineTo(proj.px, proj.py);
        }
        ctx.stroke();
      });

      // Normal traffic (green)
      normalParticles.forEach((p) => {
        const a = p.angle + time * p.speed;
        const px3d = Math.cos(a) * p.r;
        const py3d = p.yOff + Math.sin(time + p.angle) * 5;
        const pz3d = Math.sin(a) * p.r;
        const proj = project3D(px3d, py3d, pz3d, rotY, rotX);
        if (proj.z > -50) {
          ctx.beginPath();
          ctx.arc(proj.px, proj.py, p.size * proj.scale, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(0, 255, 65, ${0.6 * proj.scale})`;
          ctx.fill();
        }
      });

      // Attack particles (red)
      attackParticles.forEach((p) => {
        const a = p.theta + time * p.speed;
        const px3d = p.r * Math.sin(p.phi) * Math.cos(a);
        const py3d = p.r * Math.sin(p.phi) * Math.sin(a);
        const pz3d = p.r * Math.cos(p.phi);
        const proj = project3D(px3d, py3d, pz3d, rotY * 0.5, rotX);
        if (proj.z > -60) {
          ctx.beginPath();
          ctx.arc(proj.px, proj.py, p.size * proj.scale, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 0, 60, ${0.5 * proj.scale})`;
          ctx.shadowColor = '#FF003C';
          ctx.shadowBlur = 4;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      });

      // Scan line
      const scanY = cy + Math.sin(time * 1.5) * 80;
      const scanGrad = ctx.createLinearGradient(cx - 120, scanY, cx + 120, scanY);
      scanGrad.addColorStop(0, 'transparent');
      scanGrad.addColorStop(0.5, 'rgba(0, 240, 255, 0.1)');
      scanGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = scanGrad;
      ctx.fillRect(cx - 120, scanY - 1, 240, 2);

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
