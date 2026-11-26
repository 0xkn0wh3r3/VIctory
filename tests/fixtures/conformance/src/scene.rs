//! The GPU frame's scene: a ground plane,
//! two cubes and a glowing pyramid, lit by one directional light, textured
//! from a generated checkerboard, then blurred by a compute pass. Plain
//! data and arithmetic, shared by the conformance title (which draws it
//! through VIctory) and its harness (which draws it natively on WARP from
//! equivalent HLSL), so both draw exactly the same thing.

/// The frame's size in pixels.
pub const WIDTH: u32 = 512;
pub const HEIGHT: u32 = 288;
/// The compute pass's thread group (8 x 8 pixels).
pub const GROUP: u32 = 8;
/// The texture's size in texels (RGBA8).
pub const TEXTURE: u32 = 64;

/// RGBA8 values, each a whole number of 255ths so that they survive the
/// render target's conversion exactly.
pub const CLEAR: [u8; 4] = [40, 48, 64, 255];
/// The pyramid's emitted colour (it reflects nothing).
pub const GLOW: [u8; 4] = [255, 170, 40, 255];
/// Pixels of the checked frame: one inside the pyramid where it is
/// nearest (4 pixels from any edge), and one inside its outline where the
/// nearer cube (drawn after it) hides it.
pub const GLOW_PIXEL: (u32, u32) = (252, 143);
pub const HIDDEN_PIXEL: (u32, u32) = (235, 150);

/// A vertex as the input layout describes it: position (`POSITION`, three
/// floats at 0), normal (`NORMAL`, three at 12), texture coordinates
/// (`TEXCOORD`, two at 24).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}
pub const STRIDE: u32 = 32;

/// A draw: its indices in the index buffer and its first vertex.
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub first_index: u32,
    pub indices: u32,
    pub base_vertex: i32,
}

/// All meshes in one vertex and one index buffer (16-bit indices).
pub struct Geometry {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
    /// Ground, cube, pyramid; then, drawn only in the demo's window (never
    /// in the checked frame), the terrain, the sky's backdrop, the floor and
    /// the highway.
    pub meshes: [Draw; 7],
}

type V3 = [f32; 3];

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Screen right as seen by a viewer looking at a face with outward normal
/// `n` (left-handed: x right, y up, z into the screen), and screen up.
fn face_axes(n: V3) -> (V3, V3) {
    let up = if n[1].abs() > 0.5 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    (cross(up, scale(n, -1.0)), up)
}

impl Geometry {
    fn push_quad(&mut self, centre: V3, n: V3, half: f32, uv_scale: f32) {
        let (right, up) = face_axes(n);
        let base = self.vertices.len() as u16;
        // Top left, top right, bottom right, bottom left: clockwise as seen
        // from outside, the front face in D3D's default winding.
        for (a, b) in [(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
            let p = add(centre, add(scale(right, a * half), scale(up, b * half)));
            self.vertices.push(Vertex {
                position: p,
                normal: n,
                uv: [(a + 1.0) * 0.5 * uv_scale, (1.0 - b) * 0.5 * uv_scale],
            });
        }
        self.indices.extend([0, 1, 2, 0, 2, 3].map(|i| base + i));
    }

    /// Hills and mountains around the ground: a heightfield of Perlin
    /// noise over x in -48..48 and z in -12..70, flat (just under the
    /// ground) beneath and beside it, rising with distance from its edges
    /// into a valley of ridges.
    fn push_terrain(&mut self) {
        const COLUMNS: u16 = 160;
        const ROWS: u16 = 128;
        let (x0, x1, z0, z1) = (-48.0f32, 48.0f32, -12.0f32, 70.0f32);
        let noise = Perlin::new(0x5eed);
        let height = |x: f32, z: f32| {
            // How far beyond the 16 x 16 ground's far and side edges.
            let away = (z - 8.0).max(x.abs() - 8.0).max(0.0);
            let rise = smoothstep(0.0, 18.0, away);
            // Low in the middle, so the sun shows between the ridges.
            let valley = 0.22 + 0.78 * smoothstep(5.0, 30.0, x.abs());
            let peaks = (2.0 + 11.0 * (away / 55.0).min(1.0)) * valley;
            -0.02 + rise * peaks * noise.ridged(x * 0.045, z * 0.045, 5).powf(1.6)
        };
        let base = self.vertices.len() as u16;
        let (dx, dz) = ((x1 - x0) / COLUMNS as f32, (z1 - z0) / ROWS as f32);
        for row in 0..=ROWS {
            for column in 0..=COLUMNS {
                let (x, z) = (x0 + column as f32 * dx, z0 + row as f32 * dz);
                // The normal from the height's slope across one cell.
                let sx = height(x + dx, z) - height(x - dx, z);
                let sz = height(x, z + dz) - height(x, z - dz);
                self.vertices.push(Vertex {
                    position: [x, height(x, z), z],
                    normal: normalize([-sx, 2.0 * dx.min(dz), -sz]),
                    // The grid texture across it, a cell to a unit.
                    uv: [x / 4.0, z / 4.0],
                });
            }
        }
        let at = |column: u16, row: u16| base + row * (COLUMNS + 1) + column;
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                // As the ground's quad: far left, far right, near right, near
                // left (clockwise seen from above).
                let (a, b) = (at(column, row + 1), at(column + 1, row + 1));
                let (c, d) = (at(column + 1, row), at(column, row));
                self.indices.extend([a, b, c, a, c, d]);
            }
        }
    }

    /// The sky behind everything: a quad far down the view, facing it, whose
    /// texture is the sky and the sun ([`sky_texels`]). Its normal is the
    /// window's light, so that with [`backdrop`]'s constants it shows its
    /// texture as it is.
    fn push_backdrop(&mut self) {
        let base = self.vertices.len() as u16;
        let (x, top, bottom, z) = BACKDROP;
        let normal = synth_light();
        for (p, uv) in [
            ([-x, top, z], [0.001, 0.001]),
            ([x, top, z], [0.999, 0.001]),
            ([x, bottom, z], [0.999, 0.999]),
            ([-x, bottom, z], [0.001, 0.999]),
        ] {
            self.vertices.push(Vertex {
                position: p,
                normal,
                uv,
            });
        }
        // Top left, top right, bottom right, bottom left: clockwise as the
        // camera sees it.
        self.indices.extend([0, 1, 2, 0, 2, 3].map(|i| base + i));
    }

    /// The highway: [`ROAD_WIDTH`] wide and [`ROAD_LENGTH`] long from z = 0,
    /// just above the floor. Across it the grid texture's one cell, so its
    /// lines run along both edges; along it a line every [`ROAD_PERIOD`].
    fn push_road(&mut self) {
        let base = self.vertices.len() as u16;
        let (x, z) = (ROAD_WIDTH / 2.0, ROAD_LENGTH);
        let v = z / ROAD_PERIOD * 0.25;
        for (p, uv) in [
            ([-x, 0.01, z], [0.0, 0.0]),
            ([x, 0.01, z], [0.25, 0.0]),
            ([x, 0.01, 0.0], [0.25, v]),
            ([-x, 0.01, 0.0], [0.0, v]),
        ] {
            self.vertices.push(Vertex {
                position: p,
                normal: [0.0, 1.0, 0.0],
                uv,
            });
        }
        // As the ground's quad: far left, far right, near right, near left.
        self.indices.extend([0, 1, 2, 0, 2, 3].map(|i| base + i));
    }

    fn mesh(&mut self, build: impl FnOnce(&mut Geometry)) -> Draw {
        let first_index = self.indices.len() as u32;
        let base_vertex = self.vertices.len() as i32;
        let mut part = Geometry {
            vertices: Vec::new(),
            indices: Vec::new(),
            meshes: self.meshes,
        };
        build(&mut part);
        self.vertices.extend(part.vertices);
        self.indices.extend(&part.indices);
        Draw {
            first_index,
            indices: part.indices.len() as u32,
            base_vertex,
        }
    }

    pub fn new() -> Geometry {
        let none = Draw {
            first_index: 0,
            indices: 0,
            base_vertex: 0,
        };
        let mut g = Geometry {
            vertices: Vec::new(),
            indices: Vec::new(),
            meshes: [none; 7],
        };
        // A 16 x 16 ground, the texture repeated 4 times across it.
        let ground = g.mesh(|g| g.push_quad([0.0; 3], [0.0, 1.0, 0.0], 8.0, 4.0));
        // A unit cube about its centre.
        let cube = g.mesh(|g| {
            for n in [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ] {
                g.push_quad(scale(n, 0.5), n, 0.5, 1.0);
            }
        });
        // A pyramid of height 1 on a unit square, about its centre.
        let pyramid = g.mesh(|g| {
            g.push_quad([0.0, -0.5, 0.0], [0.0, -1.0, 0.0], 0.5, 1.0);
            for d in [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ] {
                let (right, _) = face_axes(d);
                let n = normalize(add(scale(d, 2.0), [0.0, 1.0, 0.0]));
                let base = g.vertices.len() as u16;
                let corner = |side: f32| {
                    add(
                        add(scale(d, 0.5), scale(right, side * 0.5)),
                        [0.0, -0.5, 0.0],
                    )
                };
                for (p, uv) in [
                    ([0.0, 0.5, 0.0], [0.5, 0.0]),
                    (corner(1.0), [1.0, 1.0]),
                    (corner(-1.0), [0.0, 1.0]),
                ] {
                    g.vertices.push(Vertex {
                        position: p,
                        normal: n,
                        uv,
                    });
                }
                g.indices.extend([base, base + 1, base + 2]);
            }
        });
        let terrain = g.mesh(|g| g.push_terrain());
        let backdrop = g.mesh(|g| g.push_backdrop());
        // The window's floor: 128 x 128, a repeat of the grid texture every
        // 4 units (a cell a unit), as the terrain's.
        let floor = g.mesh(|g| g.push_quad([0.0; 3], [0.0, 1.0, 0.0], 64.0, 32.0));
        let road = g.mesh(|g| g.push_road());
        g.meshes = [ground, cube, pyramid, terrain, backdrop, floor, road];
        g
    }
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry::new()
    }
}

/// A 4 x 4 matrix as four rows; a point p transforms to (row i . p).
pub type Matrix = [[f32; 4]; 4];

fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let mut m = [[0.0; 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

/// Placement: uniform scale, then a turn about y (radians), then a move.
fn world(at: V3, size: f32, turn: f32) -> Matrix {
    let (s, c) = turn.sin_cos();
    [
        [c * size, 0.0, s * size, at[0]],
        [0.0, size, 0.0, at[1]],
        [-s * size, 0.0, c * size, at[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Camera: looking from `eye` at `target`, 60 degrees vertically, depth 0
/// at 0.1 and 1 at 100 (D3D's clip space).
fn view_projection(eye: V3, target: V3) -> Matrix {
    let z = normalize(add(target, scale(eye, -1.0)));
    let x = normalize(cross([0.0, 1.0, 0.0], z));
    let y = cross(z, x);
    let view = [
        [x[0], x[1], x[2], -dot(x, eye)],
        [y[0], y[1], y[2], -dot(y, eye)],
        [z[0], z[1], z[2], -dot(z, eye)],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let (near, far) = (0.1f32, 100.0f32);
    let ys = 1.0 / (30.0f32.to_radians()).tan();
    let xs = ys * HEIGHT as f32 / WIDTH as f32;
    let q = far / (far - near);
    let projection = [
        [xs, 0.0, 0.0, 0.0],
        [0.0, ys, 0.0, 0.0],
        [0.0, 0.0, q, -near * q],
        [0.0, 0.0, 1.0, 0.0],
    ];
    multiply(&projection, &view)
}

/// The per-frame constants (`b0`), 256 bytes: the view-projection rows,
/// then (each a float4) the direction towards the light, the eye, the
/// light's colour and the ambient light.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SceneConstants {
    pub view_projection: Matrix,
    pub light: [f32; 4],
    pub eye: [f32; 4],
    pub light_colour: [f32; 4],
    pub ambient: [f32; 4],
    pub _pad: [f32; 32],
}

/// The per-draw constants (`b1`), 256 bytes: the world rows, then the
/// albedo with the specular exponent in w, the specular colour and the
/// emitted colour.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MeshConstants {
    pub world: Matrix,
    pub albedo_shininess: [f32; 4],
    pub specular: [f32; 4],
    pub emissive: [f32; 4],
    pub _pad: [f32; 36],
}

const _: () = assert!(size_of::<SceneConstants>() == 256);
const _: () = assert!(size_of::<MeshConstants>() == 256);

/// The frame at time `t` seconds (0 for the checked frame): its constants
/// and, per draw, which mesh it draws and with what constants. Draw order:
/// ground, pyramid, then the cubes (the nearer cube hides part of the
/// pyramid, drawn before it: the depth test decides).
pub fn frame(t: f32) -> (SceneConstants, Vec<(usize, MeshConstants)>) {
    let eye = [0.0, 2.6, -4.6];
    let light = normalize([-0.45, 0.8, -0.4]);
    let scene = SceneConstants {
        view_projection: view_projection(eye, [0.1, 0.55, 0.7]),
        light: [light[0], light[1], light[2], 0.0],
        eye: [eye[0], eye[1], eye[2], 1.0],
        light_colour: [1.0, 0.96, 0.9, 0.0],
        ambient: [0.18, 0.2, 0.26, 0.0],
        _pad: [0.0; 32],
    };
    let mesh = |world, albedo: V3, shininess, specular: V3, emissive: V3| MeshConstants {
        world,
        albedo_shininess: [albedo[0], albedo[1], albedo[2], shininess],
        specular: [specular[0], specular[1], specular[2], 0.0],
        emissive: [emissive[0], emissive[1], emissive[2], 0.0],
        _pad: [0.0; 36],
    };
    let glow = GLOW.map(|c| f32::from(c) / 255.0);
    let draws = vec![
        (
            0,
            mesh(
                world([0.0; 3], 1.0, 0.0),
                [0.9; 3],
                16.0,
                [0.15; 3],
                [0.0; 3],
            ),
        ),
        (
            2,
            mesh(
                world([0.05, 0.55, 1.6], 1.1, 0.4 + t * 0.7),
                [0.0; 3],
                1.0,
                [0.0; 3],
                [glow[0], glow[1], glow[2]],
            ),
        ),
        (
            1,
            mesh(
                world([-1.0, 0.6, 0.0], 1.2, 0.5 + t * 0.5),
                [0.95, 0.45, 0.35],
                48.0,
                [0.8; 3],
                [0.0; 3],
            ),
        ),
        (
            1,
            mesh(
                world([1.5, 0.5, 1.2], 1.0, -0.35 - t * 0.9),
                [0.35, 0.6, 0.95],
                24.0,
                [0.5; 3],
                [0.0; 3],
            ),
        ),
    ];
    (scene, draws)
}

/// The demo window's look: synthwave. Its sky texture's size in texels.
pub const SKY_WIDTH: u32 = 768;
pub const SKY_HEIGHT: u32 = 256;
/// The backdrop: half its width, its top and bottom, and how far down the
/// view it stands (95.5 from the window's eye: inside the far plane at 100).
const BACKDROP: (f32, f32, f32, f32) = (130.0, 64.0, -20.0, 88.0);
/// The sun on the backdrop: its centre's height and its radius.
const SUN: (f32, f32) = (15.0, 15.0);
/// The window's light: its colour and the ambient light.
const SYNTH_LIGHT_COLOUR: V3 = [0.85, 0.55, 0.95];
const SYNTH_AMBIENT: V3 = [0.42, 0.28, 0.62];

/// Towards the window's light: from above, in front and to the left.
fn synth_light() -> V3 {
    normalize([-0.3, 0.8, -0.5])
}

/// The demo's drive: the car's speed (units a second), the highway's size,
/// and the period along it of the roadside rows. Every pattern along the
/// highway repeats within [`WRAP`] units, where the car's distance wraps.
const SPEED: f32 = 18.0;
const ROAD_WIDTH: f32 = 6.0;
const ROAD_LENGTH: f32 = 128.0;
const ROAD_PERIOD: f32 = 8.0;
const ROADSIDE: f32 = 128.0;
const WRAP: f32 = 4096.0;

/// The window's time: `elapsed` wrapped every 65,536 / [`SPEED`] seconds
/// (an hour), when the car has gone a whole number of [`WRAP`]s, so a long
/// run keeps f32's precision.
pub fn demo_time(elapsed: std::time::Duration) -> f32 {
    (elapsed.as_secs_f64() % (65_536.0 / f64::from(SPEED))) as f32
}

/// How far the car has gone at time `t` (wrapped), and how far it is from
/// the highway's middle (it weaves between the lanes).
fn car_at(t: f32) -> (f32, f32) {
    ((SPEED * t) % WRAP, 1.6 * (0.45 * t).sin())
}

/// `v` down to a multiple of `period`: a pattern that repeats every
/// `period` moved by this looks still while the car goes on.
fn snap(v: f32, period: f32) -> f32 {
    (v / period).floor() * period
}

/// A box: the cube scaled by `size` (x, y, z) about its centre, turned by
/// `turn` about y, centred at `at`.
fn world_box(at: V3, size: V3, turn: f32) -> Matrix {
    let (s, c) = turn.sin_cos();
    [
        [c * size[0], 0.0, s * size[2], at[0]],
        [0.0, size[1], 0.0, at[1]],
        [-s * size[0], 0.0, c * size[2], at[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// The demo window's frame at time `t`: a car driving down a neon highway
/// into the sunset, the camera following behind it, past spinning shapes
/// (the stress scale's, [`crowd`]), in synthwave light and colours.
pub fn window_frame(t: f32) -> (SceneConstants, Vec<(usize, MeshConstants)>) {
    let (mut scene, _) = frame(t);
    let (z, x) = car_at(t);
    // The camera lags the car's weave a little.
    let (_, lagged) = car_at(t - 0.35);
    let eye = [lagged * 0.9, 2.3, z - 7.2];
    let light = synth_light();
    scene.view_projection = view_projection(eye, [x * 0.6, 0.9, z + 12.0]);
    scene.eye = [eye[0], eye[1], eye[2], 1.0];
    scene.light = [light[0], light[1], light[2], 0.0];
    let [r, g, b] = SYNTH_LIGHT_COLOUR;
    scene.light_colour = [r, g, b, 0.0];
    let [r, g, b] = SYNTH_AMBIENT;
    scene.ambient = [r, g, b, 0.0];
    let mesh = |world, albedo: V3, shininess, specular: V3, emissive: V3| MeshConstants {
        world,
        albedo_shininess: [albedo[0], albedo[1], albedo[2], shininess],
        specular: [specular[0], specular[1], specular[2], 0.0],
        emissive: [emissive[0], emissive[1], emissive[2], 0.0],
        _pad: [0.0; 36],
    };
    let none = [0.0; 3];
    // The floor and the highway move on in whole repeats of their texture,
    // so they seem to scroll past.
    let floor = world([0.0, 0.0, snap(z, 4.0) + 50.0], 1.0, 0.0);
    let road = world([0.0, 0.0, snap(z, ROAD_PERIOD) - 16.0], 1.0, 0.0);
    let mut draws = vec![
        (5, mesh(floor, [1.0; 3], 16.0, [0.3, 0.1, 0.4], none)),
        (6, mesh(road, [0.5, 0.6, 1.0], 32.0, [0.4, 0.2, 0.6], none)),
    ];
    // The car, facing along the highway, turned into its weave.
    let turn = (1.6 * 0.45 * (0.45 * t).cos() / SPEED).atan();
    let (s, c) = turn.sin_cos();
    let part = |l: V3| [x + c * l[0] + s * l[2], l[1], z - s * l[0] + c * l[2]];
    let car = |l: V3, size: V3| world_box(part(l), size, turn);
    let body = car([0.0, 0.45, 0.0], [1.9, 0.45, 4.2]);
    draws.push((1, mesh(body, [1.0, 0.25, 0.75], 64.0, [0.9; 3], none)));
    let cabin = car([0.0, 0.85, -0.3], [1.5, 0.4, 2.0]);
    draws.push((
        1,
        mesh(cabin, [0.15, 0.1, 0.3], 96.0, [0.8; 3], [0.05, 0.2, 0.3]),
    ));
    for (wx, wz) in [(-0.95, 1.35), (0.95, 1.35), (-0.95, -1.35), (0.95, -1.35)] {
        let wheel = car([wx, 0.22, wz], [0.3, 0.44, 0.7]);
        draws.push((1, mesh(wheel, [0.05; 3], 8.0, [0.1; 3], none)));
    }
    for lx in [-0.65, 0.65] {
        let light = car([lx, 0.5, -2.11], [0.45, 0.12, 0.04]);
        draws.push((1, mesh(light, none, 1.0, none, [1.0, 0.1, 0.25])));
    }
    // Landmarks: the checked frame's glowing pyramid and two cubes, beside
    // the highway once every [`ROADSIDE`] units.
    let landmark = |offset: f32| z + (offset - z).rem_euclid(ROADSIDE) - 12.0;
    let glow = GLOW.map(|c| f32::from(c) / 255.0);
    let pyramid = world([-6.5, 1.2, landmark(40.0)], 2.4, t * 0.7);
    draws.push((
        2,
        mesh(pyramid, none, 1.0, none, [glow[0], glow[1], glow[2]]),
    ));
    let pink = world([7.0, 1.0, landmark(70.0)], 2.0, 0.5 + t * 0.5);
    draws.push((1, mesh(pink, [1.0, 0.35, 0.85], 48.0, [0.8; 3], none)));
    let cyan = world([-7.5, 1.0, landmark(100.0)], 2.0, -0.35 - t * 0.9);
    draws.push((1, mesh(cyan, [0.25, 0.95, 1.0], 24.0, [0.5; 3], none)));
    (scene, draws)
}

/// The terrain's constants at time `t`: it travels with the car (its
/// ridges are far off, at the horizon), and its colour.
pub fn terrain(t: f32) -> MeshConstants {
    MeshConstants {
        world: world([0.0, 0.0, car_at(t).0], 1.0, 0.0),
        albedo_shininess: [0.9, 0.6, 1.0, 6.0],
        specular: [0.15, 0.05, 0.2, 0.0],
        emissive: [0.0; 4],
        _pad: [0.0; 36],
    }
}

/// The backdrop's constants at time `t`: it travels with the car (the sky
/// is at infinity), with an albedo that undoes the window's light on it
/// (its normal is the light, so the light's term is whole) and no specular:
/// it shows the sky texture as it is.
pub fn backdrop(t: f32) -> MeshConstants {
    let lit = |c: usize| 1.0 / (SYNTH_AMBIENT[c] + SYNTH_LIGHT_COLOUR[c]);
    MeshConstants {
        world: world([0.0, 0.0, car_at(t).0], 1.0, 0.0),
        albedo_shininess: [lit(0), lit(1), lit(2), 1.0],
        specular: [0.0; 4],
        emissive: [0.0; 4],
        _pad: [0.0; 36],
    }
}

/// Objects the demo's stress scale adds to the frame (none in the checked
/// frame): `count` small cubes and pyramids spinning and bobbing in rows
/// along both sides of the highway, 32 to a row every [`ROADSIDE`] units,
/// streaming past the car; each a draw with constants of its own.
pub fn crowd(t: f32, count: usize) -> Vec<(usize, MeshConstants)> {
    let (z, _) = car_at(t);
    let step = ROADSIDE / 32.0;
    (0..count)
        .map(|i| {
            let side = if i % 2 == 0 { -1.0 } else { 1.0 };
            let k = i / 2;
            let (row, along) = ((k / 32) as f32, (k % 32) as f32);
            let phase = i as f32 * 0.37;
            // Placed along the highway, then moved on by whole repeats.
            let offset = along * step + if side > 0.0 { step / 2.0 } else { 0.0 };
            let ahead = z + (offset - z).rem_euclid(ROADSIDE) - 12.0;
            let size = 0.9;
            let at = [
                side * (ROAD_WIDTH / 2.0 + 2.0 + row * 2.0),
                size * 0.5 + 0.3 * (t * 2.5 + phase).sin().abs(),
                ahead,
            ];
            let turn = phase + t * (1.5 + (i % 5) as f32 * 0.4);
            let albedo = hue(i as f32 * 0.618_034 % 1.0);
            let constants = MeshConstants {
                world: world(at, size, turn),
                albedo_shininess: [albedo[0], albedo[1], albedo[2], 24.0],
                specular: [0.4, 0.4, 0.4, 0.0],
                emissive: [0.0; 4],
                _pad: [0.0; 36],
            };
            // Cubes and pyramids alternate along each side.
            (1 + k % 2, constants)
        })
        .collect()
}

/// The sky's texels, rows of RGBA8 ([`SKY_WIDTH`] x [`SKY_HEIGHT`]): a dusk
/// gradient from indigo to hot pink at the horizon, stars, and a striped
/// sun, yellow above and pink below, with a glow about it.
pub fn sky_texels() -> Vec<u8> {
    let (half, top, bottom, _) = BACKDROP;
    let (sun_y, sun_r) = SUN;
    let mix = |a: V3, b: V3, t: f32| -> V3 {
        let t = t.clamp(0.0, 1.0);
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    };
    let mut out = Vec::with_capacity((SKY_WIDTH * SKY_HEIGHT * 4) as usize);
    for j in 0..SKY_HEIGHT {
        for i in 0..SKY_WIDTH {
            // The texel's place on the backdrop, in world units.
            let x = -half + (i as f32 + 0.5) / SKY_WIDTH as f32 * 2.0 * half;
            let y = top - (j as f32 + 0.5) / SKY_HEIGHT as f32 * (top - bottom);
            let up = y / top;
            let mut c = if y < 0.0 {
                [24.0, 6.0, 44.0]
            } else if up < 0.3 {
                mix([255.0, 70.0, 120.0], [170.0, 30.0, 140.0], up / 0.3)
            } else {
                mix([170.0, 30.0, 140.0], [18.0, 6.0, 48.0], (up - 0.3) / 0.7)
            };
            // Stars, high up: a few texels by a hash of their place.
            let hash = (i.wrapping_mul(73_856_093) ^ j.wrapping_mul(19_349_663)) % 1000;
            if up > 0.4 && hash < 3 {
                c = mix(c, [255.0, 230.0, 255.0], 0.4 + 0.2 * hash as f32);
            }
            let d = (x * x + (y - sun_y) * (y - sun_y)).sqrt();
            if d < sun_r {
                // Stripes cut across its lower half, wider towards the bottom.
                let below = sun_y - y;
                let gap = ((below - 1.5) / (sun_r - 1.5)).clamp(0.0, 1.0) * 0.6;
                let striped = below > 0.0 && (below % 2.4) / 2.4 < gap;
                if !striped {
                    c = mix(
                        [255.0, 40.0, 150.0],
                        [255.0, 225.0, 80.0],
                        (y - (sun_y - sun_r)) / (2.0 * sun_r),
                    );
                }
            } else if d < sun_r + 7.0 {
                let glow = (1.0 - (d - sun_r) / 7.0).powi(2) * 0.5;
                c = mix(c, [255.0, 80.0, 160.0], glow);
            }
            out.extend([c[0] as u8, c[1] as u8, c[2] as u8, 255]);
        }
    }
    out
}

/// The window's grid, rows of RGBA8 ([`TEXTURE`] x [`TEXTURE`], as
/// [`texels`]): dark violet with glowing magenta lines every 16 texels, so
/// four cells to a repeat of the texture.
pub fn grid_texels() -> Vec<u8> {
    let background = [16.0f32, 6.0, 34.0];
    let line = [255.0f32, 50.0, 210.0];
    let glow = |at: u32| {
        let f = (at as f32 + 0.5) % 16.0;
        let d = f.min(16.0 - f);
        if d < 1.0 {
            1.0
        } else {
            (-(d - 1.0) * 1.2).exp() * 0.6
        }
    };
    let mut out = Vec::with_capacity((TEXTURE * TEXTURE * 4) as usize);
    for y in 0..TEXTURE {
        for x in 0..TEXTURE {
            let k = glow(x).max(glow(y));
            let c: [f32; 3] =
                std::array::from_fn(|n| background[n] + (line[n] - background[n]) * k);
            out.extend([c[0] as u8, c[1] as u8, c[2] as u8, 255]);
        }
    }
    out
}

fn smoothstep(from: f32, to: f32, x: f32) -> f32 {
    let t = ((x - from) / (to - from)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Ken Perlin's improved gradient noise in two dimensions, with its
/// permutation shuffled from `seed` (the same terrain every run).
struct Perlin {
    p: [u8; 512],
}

impl Perlin {
    fn new(seed: u64) -> Perlin {
        let mut table: [u8; 256] = std::array::from_fn(|i| i as u8);
        let mut state = seed;
        for i in (1..256).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            table.swap(i, (state >> 33) as usize % (i + 1));
        }
        Perlin {
            p: std::array::from_fn(|i| table[i % 256]),
        }
    }

    /// Noise at (x, y), about -1 to 1.
    fn noise(&self, x: f32, y: f32) -> f32 {
        let (xf, yf) = (x.floor(), y.floor());
        let (xi, yi) = ((xf as i32 & 255) as usize, (yf as i32 & 255) as usize);
        let (x, y) = (x - xf, y - yf);
        let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let grad = |hash: u8, x: f32, y: f32| match hash & 7 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            3 => -x - y,
            4 => x,
            5 => -x,
            6 => y,
            _ => -y,
        };
        let p = &self.p;
        let corner = |dx: usize, dy: usize| p[p[xi + dx] as usize + yi + dy];
        let (u, v) = (fade(x), fade(y));
        let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);
        lerp(
            lerp(grad(corner(0, 0), x, y), grad(corner(1, 0), x - 1.0, y), u),
            lerp(
                grad(corner(0, 1), x, y - 1.0),
                grad(corner(1, 1), x - 1.0, y - 1.0),
                u,
            ),
            v,
        )
    }

    /// Ridged multifractal noise of `octaves` octaves, 0 to 1: sharp crests
    /// where the noise crosses zero, as mountain ridges.
    fn ridged(&self, x: f32, y: f32, octaves: u32) -> f32 {
        let (mut sum, mut amplitude, mut frequency, mut total) = (0.0, 1.0, 1.0, 0.0);
        for _ in 0..octaves {
            let ridge = 1.0 - self.noise(x * frequency, y * frequency).abs();
            sum += amplitude * ridge * ridge;
            total += amplitude;
            amplitude *= 0.5;
            frequency *= 2.0;
        }
        sum / total
    }
}

/// A saturated colour of hue `h` (0 to 1).
fn hue(h: f32) -> V3 {
    let channel = |offset: f32| {
        let k = (h * 6.0 + offset) % 6.0;
        0.9 - 0.6 * (k.min(4.0 - k).clamp(0.0, 1.0))
    };
    [channel(5.0), channel(3.0), channel(1.0)]
}

/// The texture's texels, rows of RGBA8: a checkerboard of 8-texel cells,
/// light cells and tinted dark ones.
pub fn texels() -> Vec<u8> {
    let mut out = Vec::with_capacity((TEXTURE * TEXTURE * 4) as usize);
    for y in 0..TEXTURE {
        for x in 0..TEXTURE {
            let (cx, cy) = (x / 8, y / 8);
            let texel = if (cx + cy) % 2 == 0 {
                [235, 230, 215, 255]
            } else {
                [
                    60 + 20 * (cx % 4) as u8,
                    100 + 15 * (cy % 4) as u8,
                    170,
                    255,
                ]
            };
            out.extend(texel);
        }
    }
    out
}
