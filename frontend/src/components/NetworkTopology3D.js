import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

function NetworkNode({ position, type = 'client' }) {
  const ref = useRef();
  const colors = { server: '#00F0FF', client: '#00FF41', attacker: '#FF003C', firewall: '#FAFF00' };
  const c = colors[type] || '#00F0FF';

  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y += 0.01;
      ref.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2 + position[0]) * 0.08;
    }
  });

  return (
    <mesh ref={ref} position={position}>
      {type === 'firewall' ? <octahedronGeometry args={[0.3, 0]} /> :
       type === 'server' ? <boxGeometry args={[0.35, 0.5, 0.35]} /> :
       type === 'attacker' ? <tetrahedronGeometry args={[0.25, 0]} /> :
       <sphereGeometry args={[0.2, 16, 16]} />}
      <meshStandardMaterial color={c} emissive={c} emissiveIntensity={0.5} transparent opacity={0.85} wireframe={type === 'server'} />
    </mesh>
  );
}

function Connection({ start, end, color = '#00F0FF' }) {
  const ref = useRef();
  const dotRef = useRef();
  const progress = useRef(Math.random());

  const geometry = useMemo(() => {
    const points = [new THREE.Vector3(...start), new THREE.Vector3(...end)];
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [start, end]);

  useFrame((_, delta) => {
    progress.current = (progress.current + delta * 0.4) % 1;
    if (dotRef.current) {
      const t = progress.current;
      dotRef.current.position.set(
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t + Math.sin(t * Math.PI) * 0.2,
        start[2] + (end[2] - start[2]) * t
      );
    }
  });

  return (
    <>
      <line geometry={geometry}>
        <lineBasicMaterial color={color} transparent opacity={0.15} />
      </line>
      <mesh ref={dotRef}>
        <sphereGeometry args={[0.04, 8, 8]} />
        <meshBasicMaterial color={color} />
      </mesh>
    </>
  );
}

function ParticleField() {
  const count = 150;
  const ref = useRef();
  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 14;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 8;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 14;
    }
    return pos;
  }, []);

  useFrame((state) => { if (ref.current) ref.current.rotation.y = state.clock.elapsedTime * 0.02; });

  return (
    <points ref={ref}>
      <bufferGeometry><bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} /></bufferGeometry>
      <pointsMaterial size={0.03} color="#00F0FF" transparent opacity={0.3} sizeAttenuation />
    </points>
  );
}

function Scene() {
  const nodes = [
    { pos: [0, 0.5, 0], type: 'server' },
    { pos: [-2.5, 0, 2], type: 'firewall' },
    { pos: [2.5, 0, 2], type: 'firewall' },
    { pos: [-3, -0.5, -1.5], type: 'client' },
    { pos: [-1.5, -0.5, -2], type: 'client' },
    { pos: [1.5, -0.5, -2], type: 'client' },
    { pos: [3, -0.5, -1.5], type: 'client' },
    { pos: [-4, 1, 0], type: 'attacker' },
    { pos: [4, 1.5, -0.5], type: 'attacker' },
  ];

  const connections = [
    { from: 0, to: 1, color: '#00F0FF' }, { from: 0, to: 2, color: '#00F0FF' },
    { from: 1, to: 3, color: '#00FF41' }, { from: 1, to: 4, color: '#00FF41' },
    { from: 2, to: 5, color: '#00FF41' }, { from: 2, to: 6, color: '#00FF41' },
    { from: 7, to: 1, color: '#FF003C' }, { from: 8, to: 2, color: '#FF003C' },
  ];

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[0, 5, 5]} intensity={1} color="#00F0FF" />
      <pointLight position={[-5, 3, -5]} intensity={0.5} color="#FF003C" />
      {nodes.map((n, i) => <NetworkNode key={i} position={n.pos} type={n.type} />)}
      {connections.map((c, i) => <Connection key={i} start={nodes[c.from].pos} end={nodes[c.to].pos} color={c.color} />)}
      <ParticleField />
      <gridHelper args={[20, 20, '#00F0FF', '#0a2a2a']} position={[0, -2.5, 0]} />
      <OrbitControls enableZoom={true} enablePan={false} autoRotate autoRotateSpeed={0.5} maxPolarAngle={Math.PI / 1.5} minPolarAngle={Math.PI / 4} />
    </>
  );
}

export default function NetworkTopology3D() {
  return (
    <div className="w-full h-full" style={{ minHeight: '400px' }}>
      <Canvas camera={{ position: [0, 3, 7], fov: 55 }} style={{ background: 'transparent' }} gl={{ antialias: true, alpha: true }}>
        <Scene />
      </Canvas>
    </div>
  );
}
