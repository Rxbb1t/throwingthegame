"""Generate SK_Blob -- the player character mesh, skeleton and skin weights.

Headless Blender. Nothing here is hand-modelled; every vertex is computed from the
constants below, so changing a proportion is a re-run, not a remodel.

    blender --background --factory-startup --python Tools/gen_blob_mesh.py -- <outdir>

Writes <outdir>/SK_Blob.fbx plus preview renders.

Two facts about the Blender -> Unreal FBX trip, both established by the Step 0 probe
and both easy to get wrong:

  1. Build numerically in UE centimetres AND declare unit_settings.scale_length = 0.01,
     then export with apply_unit_scale=True. Getting this wrong lands the mesh 100x out.
  2. The armature OBJECT name becomes the root bone name in UE, sitting above every bone
     declared here. It is named "Root" for that reason.

Design spec: Docs/superpowers/specs/2026-09-09-player-skeletal-rework-design.md
"""
import bpy
import bmesh
import math
import os
import sys
from mathutils import Vector, Matrix

# =============================================================================
# PROPORTIONS -- the whole design surface. Everything below is derived.
# Units are UE centimetres, z = 0 at the feet.
# =============================================================================

TOTAL_H = 97.0

# Torso: a rounded slab, not an ellipsoid.
TORSO_Z0, TORSO_Z1 = 33.0, 73.0
TORSO_W, TORSO_D = 26.0, 19.0
TORSO_BEVEL = 6.0
TORSO_BEVEL_SEGS = 4

# Head: a sphere, visibly detached.
HEAD_D = 22.0
HEAD_CZ = 86.0

# Arms. Bone chain shoulder -> elbow -> wrist; the mesh is a capsule between the
# shoulder and wrist points, so its hemispherical cap is centred ON the shoulder
# joint and pivoting leaves no gap in the torso.
ARM_D = 7.5
SHOULDER = (0.0, 16.0, 68.0)   # y is mirrored per side; 1.25x torso half-width so the
                               # arm reads as its own tube instead of a bump on the slab
ELBOW_Z = 50.0
WRIST_Z = 33.0

# Legs. DEVIATION FROM SPEC S4, deliberate: the spec's chain was hip 33 -> knee 17
# -> ankle 0. Putting the ankle at 0 centres the capsule's bottom hemisphere on the
# ground plane, so the foot sinks one radius (5 cm) BELOW it. The foot point is
# therefore lifted to z 5, which lands the mesh bottom exactly on z 0. Visible leg
# length is still 33 (torso bottom to ground); only the bone chain shortened, 16+17
# -> 14+14.
LEG_D = 10.0
HIP = (0.0, 6.0, 33.0)
KNEE_Z = 19.0
FOOT_Z = 5.0

PELVIS_Z = 33.0

# Tessellation. Budget ~1500 tris.
LIMB_RADIAL = 10
LIMB_BODY_RINGS = 6
LIMB_CAP_RINGS = 3
HEAD_RADIAL = 12
HEAD_RINGS = 8

# Skin weights: the blend band across a joint, as a fraction of the shorter
# adjacent segment. Wider = smoother curve, mushier joint.
JOINT_BLEND = 0.25

# Preview material -- MI_Blob_Mint's teal, at the spec's target shading.
SKIN_RGB = (0.06, 0.62, 0.55)
SKIN_ROUGH = 0.18
SKIN_METAL = 0.1


# =============================================================================
# Geometry helpers
# =============================================================================

def revolve(profile, radial):
    """Revolve a (radius, z) profile around the z axis.

    A profile point with radius 0 becomes a single pole vertex, which is what
    makes this build both capsules and spheres without special-casing caps.
    Returns (verts, faces) with outward-facing winding.
    """
    verts, rings = [], []
    for r, z in profile:
        if r <= 1e-9:
            rings.append([len(verts)])
            verts.append(Vector((0.0, 0.0, z)))
        else:
            ring = []
            for i in range(radial):
                a = 2.0 * math.pi * i / radial
                ring.append(len(verts))
                verts.append(Vector((r * math.cos(a), r * math.sin(a), z)))
            rings.append(ring)

    faces = []
    for upper, lower in zip(rings, rings[1:]):
        if len(upper) == 1:
            faces += [(upper[0], lower[(i + 1) % radial], lower[i]) for i in range(radial)]
        elif len(lower) == 1:
            faces += [(lower[0], upper[i], upper[(i + 1) % radial]) for i in range(radial)]
        else:
            for i in range(radial):
                j = (i + 1) % radial
                faces.append((upper[i], upper[j], lower[j], lower[i]))
    return verts, faces


def capsule_profile(z_top, z_bot, r, body_rings, cap_rings):
    """Vertical capsule: hemisphere cap centred on z_top, tube, cap on z_bot."""
    prof = []
    for i in range(cap_rings + 1):                       # top cap, pole first
        a = math.pi / 2.0 * (1.0 - i / cap_rings)
        prof.append((r * math.cos(a), z_top + r * math.sin(a)))
    for i in range(1, body_rings):                       # tube interior
        prof.append((r, z_top + (z_bot - z_top) * i / body_rings))
    for i in range(cap_rings + 1):                       # bottom cap, pole last
        a = -math.pi / 2.0 * (i / cap_rings)
        prof.append((r * math.cos(a), z_bot + r * math.sin(a)))
    return prof


def sphere_profile(cz, r, rings):
    return [(r * math.cos(math.pi / 2.0 - math.pi * i / rings),
             cz + r * math.sin(math.pi / 2.0 - math.pi * i / rings))
            for i in range(rings + 1)]


def add_geom(bm, verts, faces, offset=Vector((0, 0, 0))):
    """Append verts/faces to a bmesh. Returns the BMVerts added, in order."""
    bverts = [bm.verts.new(v + offset) for v in verts]
    bm.verts.index_update()
    for f in faces:
        try:
            bm.faces.new([bverts[i] for i in f])
        except ValueError:
            pass  # duplicate face at a pole; harmless
    return bverts


# =============================================================================
# Build
# =============================================================================

def wipe():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.armatures, bpy.data.objects,
                  bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for item in list(block):
            block.remove(item)


def build_mesh():
    """Build the body as one mesh. Returns (object, {part: [vertex indices]})."""
    bm = bmesh.new()
    parts = {}

    def record(name, bverts):
        bm.verts.index_update()
        parts[name] = [v.index for v in bverts]

    # --- torso: bevelled box -------------------------------------------------
    # bevel() deletes the source verts and returns new ones, so the torso is
    # recorded by index range afterwards rather than by holding stale references.
    cz = (TORSO_Z0 + TORSO_Z1) / 2.0
    cube = bmesh.ops.create_cube(bm, size=1.0)["verts"]
    bmesh.ops.scale(bm, vec=Vector((TORSO_D, TORSO_W, TORSO_Z1 - TORSO_Z0)), verts=cube)
    bmesh.ops.translate(bm, vec=Vector((0.0, 0.0, cz)), verts=cube)
    bmesh.ops.bevel(bm, geom=bm.edges[:] + bm.verts[:],
                    offset=TORSO_BEVEL, segments=TORSO_BEVEL_SEGS,
                    profile=0.5, affect="EDGES", clamp_overlap=True)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    parts["torso"] = list(range(len(bm.verts)))   # nothing else built yet

    # --- head ----------------------------------------------------------------
    v, f = revolve(sphere_profile(HEAD_CZ, HEAD_D / 2.0, HEAD_RINGS), HEAD_RADIAL)
    record("head", add_geom(bm, v, f))

    # --- arms ----------------------------------------------------------------
    for side, sy in (("L", -1.0), ("R", 1.0)):
        prof = capsule_profile(SHOULDER[2], WRIST_Z, ARM_D / 2.0,
                               LIMB_BODY_RINGS, LIMB_CAP_RINGS)
        v, f = revolve(prof, LIMB_RADIAL)
        record("arm" + side, add_geom(bm, v, f, Vector((0.0, sy * SHOULDER[1], 0.0))))

    # --- legs ----------------------------------------------------------------
    for side, sy in (("L", -1.0), ("R", 1.0)):
        prof = capsule_profile(HIP[2], FOOT_Z, LEG_D / 2.0,
                               LIMB_BODY_RINGS, LIMB_CAP_RINGS)
        v, f = revolve(prof, LIMB_RADIAL)
        record("leg" + side, add_geom(bm, v, f, Vector((0.0, sy * HIP[1], 0.0))))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new("BlobMesh")
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()

    obj = bpy.data.objects.new("BlobMesh", me)
    bpy.context.collection.objects.link(obj)
    return obj, parts


BONES = [
    # name,        head,                          tail,                        parent
    ("Pelvis",     (0, 0, PELVIS_Z),              (0, 0, PELVIS_Z + 6),        None),
    ("Spine",      (0, 0, TORSO_Z0),              (0, 0, TORSO_Z1),            "Pelvis"),
    ("Head",       (0, 0, TORSO_Z1 + 2),          (0, 0, TOTAL_H),             "Spine"),
    ("UpperArmL",  (0, -SHOULDER[1], SHOULDER[2]), (0, -SHOULDER[1], ELBOW_Z), "Spine"),
    ("ForeArmL",   (0, -SHOULDER[1], ELBOW_Z),    (0, -SHOULDER[1], WRIST_Z),  "UpperArmL"),
    ("UpperArmR",  (0, SHOULDER[1], SHOULDER[2]), (0, SHOULDER[1], ELBOW_Z),   "Spine"),
    ("ForeArmR",   (0, SHOULDER[1], ELBOW_Z),     (0, SHOULDER[1], WRIST_Z),   "UpperArmR"),
    ("ThighL",     (0, -HIP[1], HIP[2]),          (0, -HIP[1], KNEE_Z),        "Pelvis"),
    ("ShinL",      (0, -HIP[1], KNEE_Z),          (0, -HIP[1], FOOT_Z),        "ThighL"),
    ("ThighR",     (0, HIP[1], HIP[2]),           (0, HIP[1], KNEE_Z),         "Pelvis"),
    ("ShinR",      (0, HIP[1], KNEE_Z),           (0, HIP[1], FOOT_Z),         "ThighR"),
]


def build_armature():
    arm_data = bpy.data.armatures.new("BlobArmature")
    # This object name becomes the UE root bone -- see module docstring.
    arm_obj = bpy.data.objects.new("Root", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    for name, head, tail, parent in BONES:
        b = arm_data.edit_bones.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        if parent:
            b.parent = arm_data.edit_bones[parent]
            b.use_connect = Vector(head) == Vector(arm_data.edit_bones[parent].tail)

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def blend(z, joint_z, band):
    """1.0 fully upper bone, 0.0 fully lower bone, linear across the band."""
    if band <= 1e-6:
        return 1.0 if z >= joint_z else 0.0
    return min(1.0, max(0.0, (z - (joint_z - band / 2.0)) / band))


def skin(mesh_obj, arm_obj, parts):
    """Explicit weights. Bone heat is NOT used.

    The head is a disconnected island, where heat weighting is unreliable, and the
    torso is a rigid slab that must not deform at all. Both are hard-assigned. The
    limbs get a linear falloff whose width is a stated design parameter rather than
    whatever the solver happened to pick.
    """
    groups = {name: mesh_obj.vertex_groups.new(name=name) for name, _, _, _ in BONES}
    co = mesh_obj.data.vertices

    for i in parts["torso"]:
        groups["Spine"].add([i], 1.0, "REPLACE")
    for i in parts["head"]:
        groups["Head"].add([i], 1.0, "REPLACE")

    arm_band = JOINT_BLEND * min(SHOULDER[2] - ELBOW_Z, ELBOW_Z - WRIST_Z)
    leg_band = JOINT_BLEND * min(HIP[2] - KNEE_Z, KNEE_Z - FOOT_Z)

    for side in ("L", "R"):
        for i in parts["arm" + side]:
            w = blend(co[i].co.z, ELBOW_Z, arm_band)
            groups["UpperArm" + side].add([i], w, "REPLACE")
            groups["ForeArm" + side].add([i], 1.0 - w, "REPLACE")
        for i in parts["leg" + side]:
            w = blend(co[i].co.z, KNEE_Z, leg_band)
            groups["Thigh" + side].add([i], w, "REPLACE")
            groups["Shin" + side].add([i], 1.0 - w, "REPLACE")

    mesh_obj.parent = arm_obj
    mod = mesh_obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm_obj


def make_material():
    mat = bpy.data.materials.new("M_BlobPreview")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*SKIN_RGB, 1.0)
    bsdf.inputs["Roughness"].default_value = SKIN_ROUGH
    bsdf.inputs["Metallic"].default_value = SKIN_METAL
    return mat


# =============================================================================
# Preview rendering
# =============================================================================

def setup_render():
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.render.resolution_x = 620
    scene.render.resolution_y = 780
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("W")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.92, .93, .93, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

    key = bpy.data.objects.new("Key", bpy.data.lights.new("Key", "AREA"))
    key.data.energy = 2.2e6
    key.data.size = 300
    key.location = (-260, -220, 300)
    key.rotation_euler = (math.radians(48), 0, math.radians(-42))
    bpy.context.collection.objects.link(key)

    fill = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", "AREA"))
    fill.data.energy = 5e5
    fill.data.size = 500
    fill.location = (300, -160, 120)
    fill.rotation_euler = (math.radians(80), 0, math.radians(62))
    bpy.context.collection.objects.link(fill)

    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    cam.data.lens = 85
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    return cam


def aim_camera(cam, yaw_deg, dist=330.0, target_z=52.0):
    """yaw 0 == looking at the character's face.

    The figure faces +X (UE actor-forward) and its arms spread along Y, so the
    camera starts on +X. Starting it on -Y sights straight down the arms and
    renders a side view labelled "front".
    """
    a = math.radians(yaw_deg)
    cam.location = (dist * math.cos(a), dist * math.sin(a), target_z + 44.0)
    direction = Vector((0, 0, target_z)) - Vector(cam.location)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def pose(arm_obj, angles):
    """angles: {bone_name: pitch_degrees}.

    Blender bones point along their own local +Y, so pitch is rotation about local X.
    Rotating about Y twists the limb around its own axis and looks like nothing happened.
    """
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, 0, 0)
    for name, deg in angles.items():
        arm_obj.pose.bones[name].rotation_euler = (math.radians(deg), 0, 0)
    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")


# =============================================================================

def main():
    outdir = sys.argv[sys.argv.index("--") + 1]
    os.makedirs(outdir, exist_ok=True)

    wipe()
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 0.01

    mesh_obj, parts = build_mesh()
    arm_obj = build_armature()
    skin(mesh_obj, arm_obj, parts)
    mesh_obj.data.materials.append(make_material())

    tris = sum(len(p.vertices) - 2 for p in mesh_obj.data.polygons)
    print("GEN_VERTS:", len(mesh_obj.data.vertices))
    print("GEN_TRIS:", tris)
    print("GEN_BONES:", [b.name for b in arm_obj.data.bones])
    zs = [v.co.z for v in mesh_obj.data.vertices]
    print("GEN_HEIGHT: %.2f to %.2f" % (min(zs), max(zs)))

    cam = setup_render()
    for label, yaw in (("front", 0), ("three_quarter", 38), ("side", 90), ("back", 180)):
        aim_camera(cam, yaw)
        render_to(os.path.join(outdir, "preview_" + label))

    # Bend test: the whole point of the rig. Straight limbs prove nothing.
    pose(arm_obj, {
        "UpperArmL": -55, "ForeArmL": -50,
        "UpperArmR": 38, "ForeArmR": -28,
        "ThighL": 42, "ShinL": -60,
        "ThighR": -30, "ShinR": -18,
    })
    aim_camera(cam, 34)
    render_to(os.path.join(outdir, "preview_bend"))
    pose(arm_obj, {})

    fbx = os.path.join(outdir, "SK_Blob.fbx")
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.export_scene.fbx(
        filepath=fbx,
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        axis_forward="-Y",
        axis_up="Z",
        add_leaf_bones=False,
        use_armature_deform_only=True,
        bake_anim=False,
        mesh_smooth_type="FACE",
        path_mode="COPY",
    )
    print("GEN_FBX:", fbx, os.path.getsize(fbx))


main()
