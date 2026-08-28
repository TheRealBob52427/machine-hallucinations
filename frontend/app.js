/**
 * app.js — "Data Sculpture" renderer (Three.js).
 *
 * The backend hands us a sequence of pre-rendered latent-walk frames plus a
 * per-frame ENERGY envelope. We re-materialise the frames as a GPU particle
 * field: every texel becomes one luminous particle. Between frames, a
 * simplex-noise flow field advects the particles while colour dissolves from
 * frame A into frame B along an organic staggered front. When the energy
 * envelope peaks (mid-morph), turbulence surges — the image disintegrates
 * into flow, then re-crystallises. That rhythm *is* the Anadol look.
 */

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

/* ------------------------------------------------------------------ */
/* Tunables                                                             */
/* ------------------------------------------------------------------ */
const PLANE_H = 2.4;                 // world height of the particle plane
const CAMERA_Z = 2.6;
const PRELOAD_AHEAD = 45;            // frames buffered ahead of the playhead
const KEEP_BEHIND = 12;              // frames retained behind the playhead
const DENSITY_PRESETS = { '1': 96, '2': 192, '3': 320 };  // particles per side

const state = {
  manifest: null, streamer: null,
  renderer: null, scene: null, camera: null, composer: null, bloomPass: null,
  material: null, points: null, clock: new THREE.Clock(),
  gridN: DENSITY_PRESETS['2'],
  playhead: 0, playing: true, bloomOn: true, turbulence: 1.0,
  mouse: new THREE.Vector3(999, 999, 0), mouseTarget: 0, mouseStrength: 0,
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ */
/* Boot + API                                                          */
/* ------------------------------------------------------------------ */
async function fetchManifest() {
  const r = await fetch('/api/manifest');
  if (!r.ok) throw new Error((await r.json()).detail || 'no manifest');
  return r.json();
}

async function generateAndWait() {
  $('btn-generate').disabled = true;
  $('status').textContent = 'LAUNCHING RENDER…';
  const r = await fetch('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  if (!r.ok) { $('status').textContent = (await r.json()).detail || 'error'; $('btn-generate').disabled = false; return; }
  const { status_url } = await r.json();
  $('progress').classList.add('active');

  // Poll the job until the walk is fully rendered, then reload the scene.
  for (;;) {
    await new Promise(res => setTimeout(res, 2000));
    const job = await (await fetch(status_url)).json();
    $('status').textContent = (job.message || job.status).toUpperCase();
    $('progress-fill').style.width = `${Math.round((job.progress || 0) * 100)}%`;
    if (job.status === 'done')   { location.reload(); return; }
    if (job.status === 'failed') {
      $('status').textContent = `FAILED — ${job.error}`;
      $('btn-generate').disabled = false;
      return;
    }
  }
}

/* ------------------------------------------------------------------ */
/* FrameStreamer — a rolling ring-buffer of THREE.Textures              */
/* (a 480-frame walk would be ~2 GB as textures; we only keep a window) */
/* ------------------------------------------------------------------ */
class FrameStreamer {
  constructor(manifest) {
    this.m = manifest;
    this.count = manifest.frame_count;
    this.loader = new THREE.TextureLoader();
    this.frames = new Map();   // idx → THREE.Texture
    this.pending = new Set();  // idx currently in flight
  }
  idx(i) { return ((i % this.count) + this.count) % this.count; }
  url(i) {
    const no = String(this.idx(i) + 1);                     // files are 1-based
    const name = this.m.file_pattern.replace(/%0(\d+)d/, (_, w) => no.padStart(+w, '0'));
    return `${this.m.base_path}/${name}`;
  }
  get(i) {
    const k = this.idx(i);
    if (this.frames.has(k)) return this.frames.get(k);
    this.load(k);
    return null;
  }
  load(k) {
    if (this.pending.has(k) || this.frames.has(k)) return;
    this.pending.add(k);
    this.loader.load(this.url(k), (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;                // decode sRGB → linear
      tex.minFilter = tex.magFilter = THREE.LinearFilter;
      tex.generateMipmaps = false;
      this.frames.set(k, tex);
      this.pending.delete(k);
    }, undefined, () => this.pending.delete(k));
  }
  /** Keep [center-KEEP_BEHIND, center+PRELOAD_AHEAD] resident; evict the rest. */
  update(center) {
    for (let d = -KEEP_BEHIND; d <= PRELOAD_AHEAD; d++) this.get(center + d);
    const n = this.count;
    for (const [k, tex] of this.frames) {
      const fwd = ((k - center) % n + n) % n;               // distance ahead of center
      const inWindow = fwd <= PRELOAD_AHEAD + 8 || fwd >= n - KEEP_BEHIND - 8;
      if (!inWindow) { tex.dispose(); this.frames.delete(k); }
    }
  }
  bufferedAhead(center) {
    let n = 0;
    while (n < PRELOAD_AHEAD && this.frames.has(this.idx(center + n))) n++;
    return n;
  }
}

/* ------------------------------------------------------------------ */
/* Shaders                                                              */
/* ------------------------------------------------------------------ */
const VERT = /* glsl */`
uniform sampler2D uTexA, uTexB;
uniform float uMix;          // 0..1 cross-frame blend
uniform float uEnergy;       // 0 = crystallised … 1 = full morph (backend envelope)
uniform float uTime;
uniform float uTurbulence;   // user scalar ([+]/[-] keys)
uniform float uPointScale;
uniform vec2  uMouse;
uniform float uMouseStrength;

attribute vec2 aUv;          // this particle's texel in the frame
attribute float aRand;       // per-particle randomness

varying vec3 vColor;
varying float vAlpha;

/* --- Ashima Arts / Ian McEwan 3D simplex noise (canonical implementation) --- */
vec3 mod289(vec3 x){return x - floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x - floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0); const vec4 D = vec4(0.0,0.5,1.0,2.0);
  vec3 i = floor(v + dot(v, C.yyy)); vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz); vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy); vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx; vec3 x2 = x0 - i2 + C.yyy; vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(i.z + vec4(0.0,i1.z,i2.z,1.0))
                                   + i.y + vec4(0.0,i1.y,i2.y,1.0))
                                   + i.x + vec4(0.0,i1.x,i2.x,1.0));
  float n_ = 0.142857142857; vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z); vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy; vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy); vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0)*2.0 + 1.0; vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy; vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
  vec3 p0 = vec3(a0.xy,h.x), p1 = vec3(a0.zw,h.y), p2 = vec3(a1.xy,h.z), p3 = vec3(a1.zw,h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}

void main() {
  vec3 cA = texture2D(uTexA, aUv).rgb;
  vec3 cB = texture2D(uTexB, aUv).rgb;

  float chaos = clamp(uEnergy * uTurbulence, 0.0, 2.5);

  /* Organic dissolve front: each particle crosses over at a slightly
     different moment, driven by noise — the morph sweeps like liquid. */
  float drift = snoise(vec3(aUv * 5.0, uTime * 0.4));
  float localMix = clamp(uMix + (aRand - 0.5) * 0.35 * chaos + drift * 0.25 * chaos, 0.0, 1.0);

  vec3 col = mix(cA, cB, localMix);
  float lum = dot(col, vec3(0.299, 0.587, 0.114));
  col = mix(vec3(lum), col, 1.3 + 0.6 * chaos);     // saturation surge mid-morph
  vColor = col;

  vec3 pos = position;                               // baked grid position

  /* Advection: 3-channel simplex flow field, amplified by chaos. */
  vec3 seed = vec3(pos.xy * 2.2, uTime * 0.12 + aRand * 6.2831);
  vec3 flow = vec3(snoise(seed), snoise(seed + 31.416), snoise(seed + 47.853));
  pos += flow * (0.012 + 0.16 * chaos);

  /* Luminance relief: bright data swells toward the viewer (data sculpture). */
  pos.z += (lum - 0.5) * (0.04 + 0.10 * chaos);

  /* Pointer interaction — a soft repulsion vortex. */
  float md = distance(pos.xy, uMouse);
  float mf = exp(-md * md * 6.0) * uMouseStrength;
  pos.xy += normalize(pos.xy - uMouse + 1e-4) * mf * 0.18;
  pos.z  += mf * 0.15;

  vec4 mv = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mv;

  float sizeMul = (0.9 + 0.8 * aRand) * (1.0 + 1.6 * chaos) * (1.0 + 1.5 * mf);
  gl_PointSize = uPointScale * sizeMul / max(0.1, -mv.z);
  vAlpha = 0.5 + 0.5 * lum;
}
`;

const FRAG = /* glsl */`
varying vec3 vColor;
varying float vAlpha;
void main() {
  /* Soft round sprite, premultiplied for pure additive accumulation. */
  float d = length(gl_PointCoord - 0.5);
  float a = vAlpha * smoothstep(0.5, 0.08, d);
  gl_FragColor = vec4(vColor * a, a);
}
`;

/* ------------------------------------------------------------------ */
/* Scene                                                                */
/* ------------------------------------------------------------------ */
function initThree() {
  state.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
  state.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  state.renderer.setSize(innerWidth, innerHeight);
  state.renderer.toneMapping = THREE.ACESFilmicToneMapping;
  state.renderer.toneMappingExposure = 1.1;
  document.body.appendChild(state.renderer.domElement);

  state.scene = new THREE.Scene();
  state.scene.background = new THREE.Color(0x000000);
  state.camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 100);
  state.camera.position.set(0, 0, CAMERA_Z);

  // Bloom is 50% of the Anadol aesthetic: additive particles + glow.
  state.composer = new EffectComposer(state.renderer);
  state.composer.addPass(new RenderPass(state.scene, state.camera));
  state.bloomPass = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.85, 0.55, 0.12);
  state.composer.addPass(state.bloomPass);
  state.composer.addPass(new OutputPass());

  state.material = new THREE.ShaderMaterial({
    uniforms: {
      uTexA: { value: null }, uTexB: { value: null },
      uMix: { value: 0 }, uEnergy: { value: 0 }, uTime: { value: 0 },
      uTurbulence: { value: 1 }, uPointScale: { value: 1 },
      uMouse: { value: new THREE.Vector2(999, 999) }, uMouseStrength: { value: 0 },
    },
    vertexShader: VERT, fragmentShader: FRAG,
    transparent: true, depthTest: false, depthWrite: false,
    blending: THREE.CustomBlending,                       // pure additive
    blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
  });
}

function buildParticles(n) {
  if (state.points) {
    state.scene.remove(state.points);
    state.points.geometry.dispose();
  }
  const aspect = state.manifest ? state.manifest.width / state.manifest.height : 1;
  const W = PLANE_H * aspect, H = PLANE_H;
  const count = n * n;
  const pos = new Float32Array(count * 3);
  const uv = new Float32Array(count * 2);
  const rnd = new Float32Array(count);
  let k = 0;
  for (let iy = 0; iy < n; iy++) {
    for (let ix = 0; ix < n; ix++, k++) {
      pos[k * 3]     = ((ix + 0.5) / n - 0.5) * W;
      pos[k * 3 + 1] = ((iy + 0.5) / n - 0.5) * H;
      pos[k * 3 + 2] = 0;
      uv[k * 2] = (ix + 0.5) / n;
      uv[k * 2 + 1] = (iy + 0.5) / n;
      rnd[k] = Math.random();
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('aUv', new THREE.BufferAttribute(uv, 2));
  geo.setAttribute('aRand', new THREE.BufferAttribute(rnd, 1));
  state.gridN = n;
  state.points = new THREE.Points(geo, state.material);
  state.points.frustumCulled = false;                     // vertices are displaced
  state.scene.add(state.points);
  updatePointScale();
}

/** Convert a world-space particle diameter → device pixels (keeps density
 *  visually constant across resolutions and density presets). */
function updatePointScale() {
  const buf = new THREE.Vector2();
  state.renderer.getDrawingBufferSize(buf);
  const worldSize = (PLANE_H / state.gridN) * 2.1;        // slight overlap → continuous surface
  const projH = 2 * Math.tan(THREE.MathUtils.degToRad(state.camera.fov * 0.5));
  state.material.uniforms.uPointScale.value = worldSize * buf.y / projH;
}

/* ------------------------------------------------------------------ */
/* Input                                                                */
/* ------------------------------------------------------------------ */
function initInput() {
  addEventListener('resize', () => {
    state.camera.aspect = innerWidth / innerHeight;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(innerWidth, innerHeight);
    state.composer.setSize(innerWidth, innerHeight);
    updatePointScale();
  });

  const ray = new THREE.Raycaster();
  const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  let lastMove = 0;
  addEventListener('pointermove', (e) => {
    const ndc = new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    ray.setFromCamera(ndc, state.camera);
    if (ray.ray.intersectPlane(plane, state.mouse)) lastMove = performance.now();
  });
  state._mouseIdle = () => performance.now() - lastMove > 250;

  addEventListener('keydown', (e) => {
    switch (e.key) {
      case ' ': state.playing = !state.playing; break;
      case 'b': state.bloomOn = !state.bloomOn; break;
      case 'h': $('hud').classList.toggle('hidden'); break;
      case '+': case '=': state.turbulence = Math.min(2.5, state.turbulence + 0.1); break;
      case '-': case '_': state.turbulence = Math.max(0.0, state.turbulence - 0.1); break;
      case 'g': $('overlay').classList.remove('hidden'); $('gen-controls').classList.remove('hidden'); break;
      default:
        if (DENSITY_PRESETS[e.key]) buildParticles(DENSITY_PRESETS[e.key]);
    }
  });
}

/* ------------------------------------------------------------------ */
/* Main loop                                                            */
/* ------------------------------------------------------------------ */
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(state.clock.getDelta(), 0.1);
  const m = state.manifest, s = state.streamer;
  const uni = state.material.uniforms;
  uni.uTime.value += dt;
  uni.uTurbulence.value = state.turbulence;

  if (m && s) {
    const N = m.frame_count;
    const i0 = Math.floor(state.playhead), f = state.playhead - i0;
    const texA = s.get(i0), texB = s.get(i0 + 1);
    s.update(i0);

    if (texA && texB) {                                   // both ready → advance
      uni.uTexA.value = texA; uni.uTexB.value = texB;
      uni.uMix.value = f;
      if (state.playing) state.playhead = (state.playhead + dt * m.fps) % N;

      // Turbulence follows the backend's latent-motion envelope.
      const e = m.energy?.length === N
        ? THREE.MathUtils.lerp(m.energy[s.idx(i0)], m.energy[s.idx(i0 + 1)], f)
        : 0.4;
      uni.uEnergy.value = e;

      $('hud-stats').textContent =
        `${m.id} · frame ${s.idx(i0) + 1}/${N} · ${m.fps} fps · buf ${s.bufferedAhead(i0)}`;
    } /* else: playhead stalls until the network catches up — no stutter */
  }

  // Gentle idle camera drift (slow orbit + breathing) — the "sculpture" feel.
  const t = uni.uTime.value;
  state.camera.position.set(Math.sin(t * 0.10) * 0.35, Math.cos(t * 0.13) * 0.22, CAMERA_Z);
  state.camera.lookAt(0, 0, 0);

  // Pointer vortex eases in while moving, decays when idle.
  state.mouseTarget = state._mouseIdle?.() ? 0 : 1;
  state.mouseStrength += (state.mouseTarget - state.mouseStrength) * 0.06;
  uni.uMouse.value.set(state.mouse.x, state.mouse.y);
  uni.uMouseStrength.value = state.mouseStrength;

  if (state.bloomOn) state.composer.render();
  else state.renderer.render(state.scene, state.camera);
}

/* ------------------------------------------------------------------ */
/* Boot                                                                 */
/* ------------------------------------------------------------------ */
async function startExperience(manifest) {
  state.manifest = manifest;
  state.streamer = new FrameStreamer(manifest);
  buildParticles(state.gridN);
  state.streamer.update(0);

  // Wait until a small runway of frames is resident before revealing.
  $('status').textContent = 'BUFFERING…';
  $('progress').classList.add('active');
  while (state.streamer.bufferedAhead(0) < 10) {
    $('progress-fill').style.width = `${state.streamer.bufferedAhead(0) * 10}%`;
    await new Promise(r => setTimeout(r, 120));
  }
  $('overlay').classList.add('hidden');
}

async function boot() {
  try { initThree(); }
  catch { $('status').textContent = 'WEBGL UNAVAILABLE'; return; }
  initInput(); animate();
  try {
    startExperience(await fetchManifest());
  } catch {
    $('status').textContent = 'NO WALK RENDERED YET';
    $('gen-controls').classList.remove('hidden');
    $('btn-generate').onclick = generateAndWait;
  }
}
boot();
