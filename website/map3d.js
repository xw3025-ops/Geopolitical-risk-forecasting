// 3D map for the MENA Risk Monitor (ink-wash palette). Countries are thin colour-coded tiles lying
// on a plain table; the hovered country lifts a little, like a card raised off
// the surface, casting a soft shadow, and its relationship threads arc out.
// Country names float just above the tiles, flat (parallel to the map plane).
// ACLED events (country view) stand on the tiles as glowing pillars.
// Exports createMap3D(container, opts) -> { setStyle, focus, setEventFilter, destroy }.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";

export const webglAvailable = () => {
  try {
    const c = document.createElement("canvas");
    return !!(window.WebGL2RenderingContext && c.getContext("webgl2"));
  } catch { return false; }
};

const BG = 0xecedeb;        // paper: grey-white
const TABLE = 0xf7f7f5;
const LAND = 0xd6d7d3;      // pale ink wash
const INK = 0x16171a;
const SLAB = 0.32;          // tile thickness: a card, not a block
const LIFT = 2.2;           // how far a hovered card rises
const LABEL_Y = 0.6;        // labels float this far above their tile

/*
 opts:
   menaFeatures   [{feature, iso}]      the countries drawn as tiles
   landFeatures   [feature]             context land (flat)
   fit            GeoJSON the camera frames
   anchors        {iso: [lon, lat]}     thread / label anchor overrides
   color(iso)     -> css colour          tile colour
   glow(iso)      -> bool                bloom on the outline
   threads(iso)   -> [{other, color, width, alert}]   drawn on hover / focus
   events         [{lon, lat, f, t, color}]  pillars (country view)
   focusIso       iso                    country view: this tile is highlighted and stays put
   labels, hideLabels [iso], labelText(iso)
   onHover(iso|null, ev), onClick(iso), onEventHover(event|null, ev)
   lanes          [{id, lon, lat, color, alert}]  shipping-lane markers (+ the Suez-Hormuz sea route)
   onLaneHover(lane|null, ev)
   inset()        -> {left, right, top, bottom} px covered by floating panels; the map is
                     centred and fitted in the remaining gap
*/
export function createMap3D(container, opts) {
  const W0 = 100;
  const width = () => container.clientWidth, height = () => container.clientHeight;

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(width(), height());
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.12;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);
  scene.fog = new THREE.Fog(BG, 140, 260);

  const camera = new THREE.PerspectiveCamera(32, width() / height(), 1, 600);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minPolarAngle = 0.2;
  controls.maxPolarAngle = 1.05;
  controls.minDistance = 40;
  controls.maxDistance = 220;

  scene.add(new THREE.AmbientLight(0xffffff, 0.62));
  scene.add(new THREE.HemisphereLight(0xffffff, 0xcfd0cc, 0.45));
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(-40, 70, 20);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -80, right: 80, top: 80, bottom: -80, near: 1, far: 220 });
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.05;  // no acne stripes on thin translucent tiles
  key.shadow.radius = 6;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xf2f4f8, 0.3);
  fill.position.set(50, 30, -40);
  scene.add(fill);

  const proj = d3.geoMercator().fitExtent([[-W0 / 2, -W0 * 0.32], [W0 / 2, W0 * 0.32]], opts.fit);
  const toXZ = (lonlat) => proj(lonlat);

  const table = new THREE.Mesh(new THREE.PlaneGeometry(600, 600), new THREE.MeshStandardMaterial({ color: TABLE, roughness: 0.95 }));
  table.rotation.x = -Math.PI / 2;
  table.position.y = -0.02;
  table.receiveShadow = true;
  scene.add(table);
  const grid = new THREE.GridHelper(260, 52, 0xc4c5c0, 0xdcddd9);
  grid.material.transparent = true;
  grid.material.opacity = 0.4;
  scene.add(grid);

  function shapesOf(feature) {
    const polys = feature.geometry.type === "Polygon" ? [feature.geometry.coordinates] : feature.geometry.coordinates;
    const shapes = [];
    for (const poly of polys) {
      const ring = (r) => r.map((p) => { const [x, y] = toXZ(p); return new THREE.Vector2(x, -y); });
      const outer = ring(poly[0]);
      if (outer.length < 3) continue;
      const shape = new THREE.Shape(outer);
      for (const hole of poly.slice(1)) shape.holes.push(new THREE.Path(ring(hole)));
      shapes.push(shape);
    }
    return shapes;
  }
  function slab(feature, depth, material) {
    const geo = new THREE.ExtrudeGeometry(shapesOf(feature), { depth, bevelEnabled: false, curveSegments: 1 });
    const m = new THREE.Mesh(geo, material);
    m.rotation.x = -Math.PI / 2;
    m.castShadow = true;
    m.receiveShadow = true;
    return m;
  }

  const landMat = new THREE.MeshStandardMaterial({ color: LAND, roughness: 0.85 });
  for (const f of opts.landFeatures) {
    try { scene.add(slab(f, 0.12, landMat)); } catch { /* degenerate geometry */ }
  }

  // ---------------------------------------------------------------- country tiles
  const tiles = {}, meshes = [], centers = {};
  for (const { feature, iso } of opts.menaFeatures) {
    const focus = opts.focusIso === iso;
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0xffffff, roughness: 0.42, metalness: 0, transmission: 0, clearcoat: 0.6, clearcoatRoughness: 0.25,
      transparent: true, depthWrite: !(opts.focusIso && !focus),
      opacity: opts.focusIso && !focus ? 0.4 : 0.86, emissive: new THREE.Color(0x000000),
    });
    let mesh;
    try { mesh = slab(feature, SLAB, mat); } catch { continue; }
    mesh.userData.iso = iso;
    const edgeMat = new THREE.LineBasicMaterial({ color: INK, transparent: true, opacity: 0.5 });  // ink outline
    for (const sh of shapesOf(feature)) {
      const pts = sh.getPoints().map((p) => new THREE.Vector3(p.x, p.y, SLAB + 0.002));
      mesh.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), edgeMat));
    }
    scene.add(mesh);
    const [x, y] = toXZ(opts.anchors?.[iso] || d3.geoCentroid(feature));
    centers[iso] = new THREE.Vector3(x, 0, y);
    tiles[iso] = { mesh, mat, edgeMat, lift: 0, liftTarget: 0 };
    meshes.push(mesh);
  }

  function setStyle() {
    for (const iso in tiles) {
      const p = tiles[iso];
      let col = new THREE.Color(opts.color(iso)).lerp(new THREE.Color(0xeeeeec), 0.06);  // slightly softened against the paper
      if (opts.focusIso && iso !== opts.focusIso) col = col.lerp(new THREE.Color(0xd9dad6), 0.6);  // neighbours recede into the wash
      p.mat.color.copy(col);
      const glow = opts.glow(iso);
      p.mat.emissive.copy(col);  // self-lit, so the risk colours stay true on the dark field
      const dimmed = opts.focusIso && iso !== opts.focusIso;
      p.mat.emissiveIntensity = dimmed ? 0.05 : 0.14;
      p.edgeMat.color.copy(glow ? col.clone().multiplyScalar(1.6) : new THREE.Color(INK));  // top countries: soft glowing outline
      p.edgeMat.opacity = glow ? 0.9 : dimmed ? 0.2 : 0.32;
    }
  }
  setStyle();

  // ---------------------------------------------------------------- flat labels
  const labels = {};
  function labelMesh(text, main) {
    const fs = 64, pad = 18;
    const c = document.createElement("canvas");
    const ctx = c.getContext("2d");
    const font = `${main ? 700 : 500} ${fs}px Inter, "Segoe UI", sans-serif`;
    ctx.font = font;
    const w = Math.ceil(ctx.measureText(text).width) + pad * 2;
    c.width = w; c.height = fs + pad * 2;
    ctx.font = font;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.lineWidth = 12; ctx.strokeStyle = "rgba(255,255,255,0.75)";
    ctx.strokeText(text, w / 2, c.height / 2 + 2);
    ctx.fillStyle = main ? "#0b0c0e" : "#1b1c1f";  // ink
    ctx.fillText(text, w / 2, c.height / 2 + 2);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    const hWorld = main ? 3.6 : 2.8;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(hWorld * w / c.height, hWorld),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: false, toneMapped: false }));
    m.rotation.x = -Math.PI / 2;  // lie flat, parallel to the map plane
    m.renderOrder = 10;
    return m;
  }
  if (opts.labels) {
    for (const { iso } of opts.menaFeatures) {
      if (!tiles[iso]) continue;
      const m = labelMesh(opts.labelText ? opts.labelText(iso) : iso, iso === opts.focusIso);
      m.position.copy(centers[iso]);
      m.visible = !(opts.hideLabels || []).includes(iso);
      m.userData.hidden = !m.visible;
      scene.add(m);
      labels[iso] = m;
    }
  }

  // ---------------------------------------------------------------- threads
  const threadGroup = new THREE.Group();
  scene.add(threadGroup);
  const pulses = [];
  const topY = (iso) => SLAB + tiles[iso].lift + 0.15;
  function clearThreads() {
    threadGroup.children.forEach((o) => { o.geometry?.dispose(); o.material?.dispose(); });
    threadGroup.clear();
    pulses.length = 0;
  }
  // Thread styles (opts.threadStyle):
  //   flow  - crisp screen-space lines fading from the hovered country into the partner's tone, a soft halo,
  //           and small particles flowing towards the partner
  //   dash  - marching dashes streaming towards each partner
  //   brush - ink-brush strokes: thick in the middle, tapering at both ends
  // Surge alerts in every style: brighter, thicker, with a vermilion pulse.
  const lineMats = [];
  const PAPER = new THREE.Color(0xf4f4f2);
  function fatLine(curve, colorsFn, widthPx, extra = {}) {
    const pts = curve.getPoints(64), pos = [], cols = [];
    pts.forEach((p, i) => { pos.push(p.x, p.y, p.z); const c = colorsFn(i / (pts.length - 1)); cols.push(c.r, c.g, c.b); });
    const g = new LineGeometry(); g.setPositions(pos); g.setColors(cols);
    const m = new LineMaterial({ vertexColors: true, linewidth: widthPx, transparent: true, worldUnits: false, toneMapped: false, ...extra });
    m.resolution.set(width(), height());
    lineMats.push(m);
    const l = new Line2(g, m); l.computeLineDistances(); l.renderOrder = 5;
    return l;
  }
  function taperedTube(curve, radius, segs = 64) {
    const geo = new THREE.TubeGeometry(curve, segs, radius, 10, false);
    const p = geo.attributes.position, v = new THREE.Vector3(), c = new THREE.Vector3();
    for (let i = 0; i <= segs; i++) {
      const u = i / segs, k = 0.12 + 0.88 * Math.pow(Math.sin(Math.PI * u), 0.75);  // brush profile
      curve.getPointAt(u, c);
      for (let j = 0; j <= 10; j++) {
        const idx = i * 11 + j;
        v.fromBufferAttribute(p, idx).sub(c).multiplyScalar(k).add(c);
        p.setXYZ(idx, v.x, v.y, v.z);
      }
    }
    p.needsUpdate = true;
    return geo;
  }
  function drawThreads(iso) {
    clearThreads();
    lineMats.length = 0;
    if (!iso || !opts.threads || !tiles[iso]) return;
    const style = opts.threadStyle || "flow";
    const a = centers[iso].clone(); a.y = topY(iso);
    for (const t of opts.threads(iso)) {
      if (!tiles[t.other]) continue;
      const b = centers[t.other].clone(); b.y = topY(t.other);
      const mid = a.clone().add(b).multiplyScalar(0.5);
      mid.y += 3 + a.distanceTo(b) * 0.32;
      const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
      const col = new THREE.Color(t.color);
      const px = 1.4 + t.width * 9 + (t.alert ? 1.5 : 0);   // t.width is 0.09-0.62 world units
      if (style === "flow") {
        // ink shadow (keeps white threads readable on the light map), glowing halo, glowing core, flowing light
        // one thin line that blooms (no halo or shadow); surge alerts glow brightest and carry the surge ball
        const core = fatLine(curve, () => col.clone().multiplyScalar(t.alert ? 1.9 : 1.45), Math.max(1.6, px * 0.6), { opacity: 1, depthTest: false });
        core.renderOrder = 7;
        threadGroup.add(core);
        if (t.alert) {  // the surge ball: its colour shows how sharp the surge is (yellow -> orange -> red)
          const sharp = new THREE.Color(opts.surgeColor ? opts.surgeColor(t.surge) : 0xff3b2f);
          const ball = new THREE.Mesh(new THREE.SphereGeometry(0.5, 18, 14), new THREE.MeshBasicMaterial({ color: sharp.multiplyScalar(2.2), toneMapped: false }));
          threadGroup.add(ball);
          pulses.push({ s: ball, curve, t0: Math.random(), speed: 0.4 });
        }
      } else if (style === "dash") {
        const l = fatLine(curve, () => col, px * 1.1, { dashed: true, dashSize: 1.1, gapSize: 0.75, dashScale: 1, opacity: 0.95 });
        l.userData.march = t.alert ? 2.6 : 1.4;
        threadGroup.add(l);
        if (t.alert) threadGroup.add(fatLine(curve, () => col, px * 3, { opacity: 0.22, vertexColors: false, color: col.getHex() }));
      } else {  // brush
        const ink = col.clone().lerp(new THREE.Color(0x15161a), 0.18);
        threadGroup.add(new THREE.Mesh(taperedTube(curve, t.width * 1.25 + (t.alert ? 0.12 : 0)),
          new THREE.MeshBasicMaterial({ color: t.alert ? ink.clone().multiplyScalar(1.6) : ink, transparent: true, opacity: 0.88, toneMapped: !t.alert })));
        if (t.alert) {
          const sp = new THREE.Mesh(new THREE.SphereGeometry(0.5, 14, 10), new THREE.MeshBasicMaterial({ color: new THREE.Color(0xff4a3d).multiplyScalar(2.6), toneMapped: false }));
          threadGroup.add(sp);
          pulses.push({ s: sp, curve, t0: 0, speed: 0.4 });
        }
      }
      const end = new THREE.Mesh(new THREE.SphereGeometry(Math.max(0.32, t.width * 1.3), 16, 12), new THREE.MeshBasicMaterial({ color: col }));
      end.position.copy(b);
      threadGroup.add(end);
    }
  }

  // ---------------------------------------------------------------- ACLED pillars
  const eventMeshes = [];
  if (opts.events?.length) {
    const hScale = d3.scaleSqrt().domain([0, d3.max(opts.events, (e) => e.f) || 1]).range([0.5, 14]);
    for (const [type, evs] of d3.group(opts.events, (e) => e.t)) {
      const color = new THREE.Color(evs[0].color);
      const geo = new THREE.CylinderGeometry(0.15, 0.15, 1, 10);
      geo.translate(0, 0.5, 0);
      const inst = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.25, roughness: 0.5 }), evs.length);
      const m4 = new THREE.Matrix4();
      evs.forEach((e, i) => {
        const [x, y] = toXZ([e.lon, e.lat]);
        const r = 1 + Math.min(2, e.f / 40);
        m4.compose(new THREE.Vector3(x, SLAB + 0.02, y), new THREE.Quaternion(), new THREE.Vector3(r, hScale(e.f), r));
        inst.setMatrixAt(i, m4);
      });
      inst.userData = { type, events: evs };
      inst.castShadow = true;
      scene.add(inst);
      eventMeshes.push(inst);
    }
  }
  const setEventFilter = (off) => eventMeshes.forEach((m) => { m.visible = !off.has(m.userData.type); });

  // ---------------------------------------------------------------- shipping lanes
  const laneGroup = new THREE.Group(), laneHits = [], laneRings = [];
  scene.add(laneGroup);
  if (opts.lanes?.length) {
    // the main sea route between Europe and the Gulf: Suez - Red Sea - Bab el-Mandeb - Arabian Sea - Hormuz - Gulf
    const route = [[32.35, 30.4], [33.4, 28.2], [35.6, 24.6], [38.6, 20.2], [41.4, 15.6], [43.35, 12.6], [45.5, 12.1],
      [49.5, 12.6], [54, 14.6], [58.6, 19.5], [59.8, 22.6], [57.6, 25.2], [56.35, 26.55], [54.2, 26.4], [51.6, 27.1], [49.6, 28.9]];
    const pts = route.map(([lo, la]) => { const [x, y] = toXZ([lo, la]); return new THREE.Vector3(x, 0.25, y); });
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(new THREE.CatmullRomCurve3(pts).getPoints(200)),
      new THREE.LineDashedMaterial({ color: 0x2f5d8a, dashSize: 0.9, gapSize: 0.6, transparent: true, opacity: 0.75 }));
    line.computeLineDistances();
    laneGroup.add(line);
    for (const L of opts.lanes) {
      const [x, y] = toXZ([L.lon, L.lat]);
      const col = new THREE.Color(L.color);
      const ring = new THREE.Mesh(new THREE.TorusGeometry(1.5, 0.18, 12, 48), new THREE.MeshBasicMaterial({ color: col.clone().multiplyScalar(L.alert ? 1.5 : 1.05), toneMapped: false }));
      ring.rotation.x = -Math.PI / 2; ring.position.set(x, 0.5, y);
      const core = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 0.5, 24), new THREE.MeshBasicMaterial({ color: col.clone().multiplyScalar(1.15), toneMapped: false }));
      core.position.set(x, 0.5, y);
      const hit = new THREE.Mesh(new THREE.CylinderGeometry(2.4, 2.4, 2, 16), new THREE.MeshBasicMaterial({ visible: false }));
      hit.position.set(x, 1, y); hit.userData.lane = L;
      laneGroup.add(ring, core, hit);
      laneHits.push(hit);
      laneRings.push({ ring, alert: L.alert, t0: Math.random() * 2 });
    }
  }
  const setLanes = (on) => { laneGroup.visible = on; };

  // ---------------------------------------------------------------- camera + post
  const target0 = opts.focusIso && centers[opts.focusIso] ? centers[opts.focusIso].clone() : new THREE.Vector3();
  function frame() {
    const W = width(), Hh = height();
    const ins = opts.inset ? opts.inset() : { left: 0, right: 0, top: 0, bottom: 0 };
    const gw = Math.max(200, W - ins.left - ins.right), gh = Math.max(200, Hh - ins.top - ins.bottom);
    // shift the projection so the scene centre sits in the middle of the free gap
    camera.setViewOffset(W, Hh, W / 2 - (ins.left + gw / 2), Hh / 2 - (ins.top + gh / 2), W, Hh);
    const tanh = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 2;
    const needW = (opts.focusIso ? 64 : 94) / (tanh * (W / Hh)) * (W / gw);
    const needH = (opts.focusIso ? 52 : 70) / tanh * (Hh / gh);
    const dist = Math.max(needW, needH);
    controls.target.copy(target0);
    camera.position.set(target0.x, dist * 0.68, target0.z + dist * 0.74);  // from due south: flat labels read level
    controls.maxDistance = dist * 2;
    controls.minDistance = dist * 0.35;
    scene.fog.near = dist * 1.3;   // fog and clip planes follow the framing distance
    scene.fog.far = dist * 2.8;
    camera.far = dist * 6;
    camera.updateProjectionMatrix();
    controls.update();
  }
  frame();
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(width(), height()), 0.45, 0.3, 0.97));  // a soft glow on HDR elements (threads, alerts, lanes)
  composer.addPass(new OutputPass());

  // ---------------------------------------------------------------- interaction
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
  let hovered = null, focused = null, down = null;
  const pick = (ev) => {
    const r = renderer.domElement.getBoundingClientRect();
    ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ptr, camera);
  };
  function raise() {
    const lifted = opts.focusIso ? null : (hovered || focused);  // country view: tiles stay flat under the pillars
    for (const k in tiles) tiles[k].liftTarget = k === lifted ? LIFT : 0;
    for (const k in labels) labels[k].visible = !labels[k].userData.hidden || k === lifted;
    if (!opts.focusIso) drawThreads(lifted);
  }
  function onMove(ev) {
    pick(ev);
    if (laneHits.length && laneGroup.visible) {
      const hit = ray.intersectObjects(laneHits, false)[0];
      if (hit) { opts.onLaneHover?.(hit.object.userData.lane, ev); renderer.domElement.style.cursor = "help"; return; }
      opts.onLaneHover?.(null, ev);
    }
    if (eventMeshes.length) {
      const hit = ray.intersectObjects(eventMeshes.filter((m) => m.visible), false)[0];
      if (hit) { opts.onEventHover?.(hit.object.userData.events[hit.instanceId], ev); renderer.domElement.style.cursor = "default"; return; }
      opts.onEventHover?.(null, ev);
    }
    const hit = ray.intersectObjects(meshes, false)[0];
    const iso = hit ? hit.object.userData.iso : null;
    renderer.domElement.style.cursor = iso ? "pointer" : "grab";
    if (iso !== hovered) { hovered = iso; raise(); }
    opts.onHover?.(iso, ev);
  }
  function onLeave(ev) { hovered = null; raise(); opts.onHover?.(null, ev); opts.onEventHover?.(null, ev); }
  const onDown = (ev) => { down = [ev.clientX, ev.clientY]; };
  function onUp(ev) {
    if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 5) return;
    pick(ev);
    const hit = ray.intersectObjects(meshes, false)[0];
    if (hit) opts.onClick?.(hit.object.userData.iso);
  }
  const el = renderer.domElement;
  el.addEventListener("pointermove", onMove);
  el.addEventListener("pointerleave", onLeave);
  el.addEventListener("pointerdown", onDown);
  el.addEventListener("pointerup", onUp);
  function focus(iso) { focused = iso; raise(); }

  const ro = new ResizeObserver(() => {
    const w = width(), h = height();
    if (!w || !h) return;
    renderer.setSize(w, h); composer.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    lineMats.forEach((m) => m.resolution.set(w, h));
    frame();
  });
  ro.observe(container);

  // ---------------------------------------------------------------- loop
  const clock = new THREE.Clock();
  let raf = 0, alive = true;
  function tick() {
    if (!alive) return;
    raf = requestAnimationFrame(tick);
    const dt = Math.min(0.05, clock.getDelta()), t = clock.elapsedTime;
    const k = 1 - Math.exp(-8.4 * dt);  // critically damped rise (DESIGN.md rate 4.2)
    let moving = false;
    for (const iso in tiles) {
      const p = tiles[iso];
      const nl = p.lift + (p.liftTarget - p.lift) * k;
      if (Math.abs(nl - p.lift) > 1e-3) moving = true;
      p.lift = nl;
      p.mesh.position.y = p.lift;
      if (labels[iso]) labels[iso].position.y = SLAB + p.lift + LABEL_Y;
    }
    if (moving && (hovered || focused) && !opts.focusIso) drawThreads(hovered || focused);
    for (const pl of pulses) pl.s.position.copy(pl.curve.getPoint((t * pl.speed + pl.t0) % 1));
    for (const o of threadGroup.children) if (o.userData.march) o.material.dashOffset = -t * o.userData.march;
    for (const r of laneRings) { const k2 = r.alert ? 1 + 0.35 * Math.sin((t + r.t0) * 3) : 1; r.ring.scale.set(k2, k2, k2); }
    controls.update();
    composer.render();
  }
  tick();
  const onVis = () => { if (document.hidden) cancelAnimationFrame(raf); else { clock.getDelta(); tick(); } };
  document.addEventListener("visibilitychange", onVis);

  function destroy() {
    alive = false;
    cancelAnimationFrame(raf);
    ro.disconnect();
    document.removeEventListener("visibilitychange", onVis);
    clearThreads();
    scene.traverse((o) => {
      o.geometry?.dispose?.();
      if (o.material) [].concat(o.material).forEach((m) => { m.map?.dispose?.(); m.dispose(); });
    });
    renderer.dispose();
    container.innerHTML = "";
  }
  return { setStyle, focus, setEventFilter, setLanes, destroy, frame };
}
