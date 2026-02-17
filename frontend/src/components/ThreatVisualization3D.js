import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

function AttackParticles({ count = 400 }) {
  const meshRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const particles = useMemo(() => {
    const p = [];
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      const radius = 2 + Math.random() * 3;
      p.push({
        position: [radius * Math.sin(phi) * Math.cos(theta), radius * Math.sin(phi) * Math.sin(theta), radius * Math.cos(phi)],
        speed: 0.2 + Math.random() * 0.5,
        offset: Math.random() * Math.PI * 2,
        scale: 0.03 + Math.random() * 0.03,
      });
    }
    return p;
  }, [count]);

  useFrame((state) => {
    if (!meshRef.current) return;
    const t = state.clock.elapsedTime;
    particles.forEach((p, i) => {
      dummy.position.set(
        p.position[0] + Math.sin(t * p.speed + p.offset) * 0.3,
        p.position[1] + Math.cos(t * p.speed + p.offset) * 0.3,
        p.position[2] + Math.sin(t * p.speed * 0.7 + p.offset) * 0.2
      );
      dummy.scale.setScalar(p.scale * (1 + Math.sin(t * 2 + p.offset) * 0.3));
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]}>
      <sphereGeometry args={[1, 6, 6]} />
      <meshBasicMaterial color="#FF003C" transparent opacity={0.7} />
    </instancedMesh>
  );
}

function NormalTrafficRing() {
  const ref = useRef();
  const count = 250;
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const particles = useMemo(() => {
    const p = [];
    for (let i = 0; i < count; i++) {
      p.push({ angle: (i / count) * Math.PI * 2, radius: 1.5 + Math.random() * 0.5, y: (Math.random() - 0.5) * 0.8, speed: 0.3 + Math.random() * 0.3 });
    }
    return p;
  }, []);

  useFrame((state) => {
    if (!ref.current) return;
    const t = state.clock.elapsedTime;
    particles.forEach((p, i) => {
      const a = p.angle + t * p.speed;
      dummy.position.set(Math.cos(a) * p.radius, p.y + Math.sin(t + p.angle) * 0.1, Math.sin(a) * p.radius);
      dummy.scale.setScalar(0.02);
      dummy.updateMatrix();
      ref.current.setMatrixAt(i, dummy.matrix);
    });
    ref.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={ref} args={[null, null, count]}>
      <sphereGeometry args={[1, 6, 6]} />
      <meshBasicMaterial color="#00FF41" transparent opacity={0.6} />
    </instancedMesh>
  );
}

function CentralShield() {
  const ref = useRef();
  const wireRef = useRef();
  useFrame((state) => {
    if (ref.current) { ref.current.rotation.y = state.clock.elapsedTime * 0.3; ref.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.2) * 0.1; }
    if (wireRef.current) wireRef.current.rotation.y = -state.clock.elapsedTime * 0.2;
  });

  return (
    <group>
      <mesh ref={ref}>
        <icosahedronGeometry args={[1, 1]} />
        <meshPhongMaterial color="#00F0FF" emissive="#00F0FF" emissiveIntensity={0.3} transparent opacity={0.15} side={THREE.DoubleSide} />
      </mesh>
      <mesh ref={wireRef}>
        <icosahedronGeometry args={[1.1, 1]} />
        <meshBasicMaterial color="#00F0FF" wireframe transparent opacity={0.3} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.4, 32, 32]} />
        <meshPhongMaterial color="#00F0FF" emissive="#00F0FF" emissiveIntensity={0.8} transparent opacity={0.6} />
      </mesh>
    </group>
  );
}

function ThreatRings() {
  const ring1 = useRef(); const ring2 = useRef(); const ring3 = useRef();
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (ring1.current) ring1.current.rotation.z = t * 0.5;
    if (ring2.current) ring2.current.rotation.z = -t * 0.3;
    if (ring3.current) ring3.current.rotation.x = t * 0.4;
  });

  return (
    <>
      <mesh ref={ring1} rotation={[Math.PI / 2, 0, 0]}><torusGeometry args={[2.5, 0.02, 8, 64]} /><meshBasicMaterial color="#FF003C" transparent opacity={0.4} /></mesh>
      <mesh ref={ring2} rotation={[Math.PI / 3, 0, 0]}><torusGeometry args={[3, 0.015, 8, 64]} /><meshBasicMaterial color="#BD00FF" transparent opacity={0.3} /></mesh>
      <mesh ref={ring3} rotation={[0, Math.PI / 4, 0]}><torusGeometry args={[3.5, 0.01, 8, 64]} /><meshBasicMaterial color="#FAFF00" transparent opacity={0.2} /></mesh>
    </>
  );
}

function Scene() {
  return (
    <>
      <ambientLight intensity={0.2} />
      <pointLight position={[5, 5, 5]} intensity={1} color="#00F0FF" />
      <pointLight position={[-5, -3, -5]} intensity={0.5} color="#FF003C" />
      <CentralShield />
      <ThreatRings />
      <AttackParticles count={350} />
      <NormalTrafficRing />
      <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.3} maxPolarAngle={Math.PI / 1.5} minPolarAngle={Math.PI / 3} />
    </>
  );
}

export default function ThreatVisualization3D() {
  return (
    <div className="w-full h-full" style={{ minHeight: '400px' }}>
      <Canvas camera={{ position: [0, 2, 6], fov: 50 }} style={{ background: 'transparent' }} gl={{ antialias: true, alpha: true }}>
        <Scene />
      </Canvas>
    </div>
  );
}
