import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Float, MeshDistortMaterial, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

const NODE_TYPES = {
  server: { color: '#00F0FF', size: 0.3, emissive: '#00F0FF' },
  client: { color: '#00FF41', size: 0.2, emissive: '#00FF41' },
  attacker: { color: '#FF003C', size: 0.25, emissive: '#FF003C' },
  firewall: { color: '#FAFF00', size: 0.35, emissive: '#FAFF00' },
};

function NetworkNode({ position, type = 'client', label }) {
  const ref = useRef();
  const config = NODE_TYPES[type];
  
  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y += 0.01;
      ref.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2 + position[0]) * 0.1;
    }
  });

  return (
    <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.5}>
      <mesh ref={ref} position={position}>
        {type === 'firewall' ? (
          <octahedronGeometry args={[config.size, 0]} />
        ) : type === 'server' ? (
          <boxGeometry args={[config.size, config.size * 1.5, config.size]} />
        ) : type === 'attacker' ? (
          <tetrahedronGeometry args={[config.size, 0]} />
        ) : (
          <sphereGeometry args={[config.size, 16, 16]} />
        )}
        <MeshDistortMaterial
          color={config.color}
          emissive={config.emissive}
          emissiveIntensity={0.5}
          distort={type === 'attacker' ? 0.4 : 0.1}
          speed={2}
          transparent
          opacity={0.85}
          wireframe={type === 'server'}
        />
      </mesh>
    </Float>
  );
}

function DataStream({ start, end, color = '#00F0FF', speed = 1 }) {
  const ref = useRef();
  const progress = useRef(0);
  
  useFrame((state, delta) => {
    progress.current = (progress.current + delta * speed * 0.5) % 1;
    if (ref.current) {
      const t = progress.current;
      ref.current.position.set(
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t + Math.sin(t * Math.PI) * 0.3,
        start[2] + (end[2] - start[2]) * t
      );
    }
  });

  return (
    <>
      <Line
        points={[start, end]}
        color={color}
        lineWidth={1}
        transparent
        opacity={0.2}
      />
      <mesh ref={ref}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.9} />
      </mesh>
    </>
  );
}

function GridFloor() {
  return (
    <gridHelper args={[20, 20, '#00F0FF', '#0a2a2a']} position={[0, -3, 0]} />
  );
}

function ParticleField() {
  const count = 200;
  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 15;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 10;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 15;
    }
    return pos;
  }, []);

  const ref = useRef();
  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y = state.clock.elapsedTime * 0.02;
    }
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial size={0.03} color="#00F0FF" transparent opacity={0.4} sizeAttenuation />
    </points>
  );
}

function Scene() {
  const nodes = [
    { pos: [0, 0.5, 0], type: 'server', label: 'Serveur' },
    { pos: [-2.5, 0, 2], type: 'firewall', label: 'Pare-feu' },
    { pos: [2.5, 0, 2], type: 'firewall', label: 'IDS' },
    { pos: [-3, -0.5, -1.5], type: 'client', label: 'Client 1' },
    { pos: [-1.5, -0.5, -2], type: 'client', label: 'Client 2' },
    { pos: [1.5, -0.5, -2], type: 'client', label: 'Client 3' },
    { pos: [3, -0.5, -1.5], type: 'client', label: 'Client 4' },
    { pos: [-4, 1, 0], type: 'attacker', label: 'Attaquant' },
    { pos: [4, 1.5, -0.5], type: 'attacker', label: 'Bot' },
  ];

  const connections = [
    { from: 0, to: 1, color: '#00F0FF' },
    { from: 0, to: 2, color: '#00F0FF' },
    { from: 1, to: 3, color: '#00FF41' },
    { from: 1, to: 4, color: '#00FF41' },
    { from: 2, to: 5, color: '#00FF41' },
    { from: 2, to: 6, color: '#00FF41' },
    { from: 7, to: 1, color: '#FF003C' },
    { from: 8, to: 2, color: '#FF003C' },
  ];

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[0, 5, 5]} intensity={1} color="#00F0FF" />
      <pointLight position={[-5, 3, -5]} intensity={0.5} color="#FF003C" />
      <pointLight position={[5, 3, -5]} intensity={0.5} color="#00FF41" />
      
      {nodes.map((node, i) => (
        <NetworkNode key={i} position={node.pos} type={node.type} label={node.label} />
      ))}
      
      {connections.map((conn, i) => (
        <DataStream
          key={i}
          start={nodes[conn.from].pos}
          end={nodes[conn.to].pos}
          color={conn.color}
          speed={conn.color === '#FF003C' ? 2 : 1}
        />
      ))}
      
      <ParticleField />
      <GridFloor />
      <OrbitControls
        enableZoom={true}
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.5}
        maxPolarAngle={Math.PI / 1.5}
        minPolarAngle={Math.PI / 4}
      />
    </>
  );
}

export default function NetworkTopology3D() {
  return (
    <div className="w-full h-full" style={{ minHeight: '400px' }}>
      <Canvas
        camera={{ position: [0, 3, 7], fov: 55 }}
        style={{ background: 'transparent' }}
        gl={{ antialias: true, alpha: true }}
      >
        <Scene />
      </Canvas>
    </div>
  );
}
