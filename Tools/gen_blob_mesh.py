"""Generate SK_Blob -- the player character mesh, skeleton and skin weights.

Headless Blender. Nothing here is hand-modelled; every vertex is computed from the
constants below, so changing a proportion is a re-run, not a remodel.

    blender --background --factory-startup --python Tools/gen_blob_mesh.py -- <outdir>

Writes <outdir>/SK_Blob.fbx plus preview renders.

THE BODY IS ONE CONTINUOUS SURFACE. The parts are built as separate primitives, then
voxel-remeshed into a single fused skin so arms and legs flow out of the torso instead
of being pills parked beside it. That fusion destroys any notion of "which part is this
vertex", so skin weights are computed from the SIGNED DISTANCE to the original analytic
primitives, which survives the remesh exactly. See assign_weights().

Three facts about the Blender -> Unreal FBX trip, all established the hard way:

  1. Build numerically in UE centimetres AND declare unit_settings.scale_length = 0.01,
     then export with apply_unit_scale=True. Getting this wrong lands the mesh 100x out.
  2. The armature OBJECT name becomes the root bone name in UE, sitting above every bone
     declared here. It is named "Root" for that reason.
  3. Blender bones point along their own local +Y. Blender's automatic roll for a bone
     pointing straight down puts local X somewhere arbitrary, so a "pitch" rotation
     twisted the legs about their own axis. Every bone here gets an EXPLICIT roll that
     puts local X on world Y, so pitch is rotation about local X for every bone alike.

Design spec: Docs/superpowers/specs/2026-09-09-player-skeletal-rework-design.md
"""
import bpy
import bmesh
import math
import os
import sys
from mathutils import Vector

# =============================================================================
# PROPORTIONS -- the whole design surface. Everything below is derived.
# Units are UE centimetres, z = 0 at the feet.
# =============================================================================

TOTAL_H = 95.0

# Torso: a rounded slab. The bevel is most of the depth, which is what stops it
# reading as a box with tidied edges.
TORSO_Z0, TORSO_Z1 = 36.0, 73.0
TORSO_W, TORSO_D = 23.0, 16.0
TORSO_BEVEL = 7.0                  # max is TORSO_D/2; at 7 of 8 the sides are near-round
TORSO_BEVEL_SEGS = 6

# Head: a sphere sitting ON the torso, not floating above it. Its underside is at
# exactly TORSO_Z1 so the remesh fuses the two into one form.
HEAD_D = 22.0
HEAD_CZ = 84.0

# Arms. Bone chain shoulder -> elbow -> wrist. The mesh capsule spans shoulder to
# wrist, so its top hemisphere is centred ON the shoulder joint and pivoting cannot
# tear a hole in the torso.
ARM_D = 7.5
SHOULDER = (0.0, 15.0, 68.0)       # BONE position, vertical. The mesh capsule leans:
ARM_TOP_Y, ARM_BOT_Y = 14.0, 18.0  # tucked into the torso at the shoulder, swinging clear
                                   # of it by the wrist -- connected at the top, a distinct
                                   # tube below, which is how the reference reads
ELBOW_Z = 52.0
WRIST_Z = 36.0

# Legs. The foot point sits one radius above the ground so the capsule's bottom cap
# lands exactly on z = 0 rather than 5 cm underneath it.
LEG_D = 10.0
HIP = (0.0, 7.0, 36.0)
KNEE_Z = 20.0
FOOT_Z = 5.0

PELVIS_Z = 36.0

# Fusion + budget.
VOXEL_SIZE = 1.1                   # at 2.2 the remesh webbed the arms to the torso and
                                   # welded the legs together; the gaps need resolving
SMOOTH_FACTOR, SMOOTH_REPEAT = 0.3, 1   # two passes at 0.5 closed the gaps again
TARGET_TRIS = 2200

# Tessellation of the source primitives (pre-remesh; only affects fusion fidelity).
LIMB_RADIAL, LIMB_BODY_RINGS, LIMB_CAP_RINGS = 12, 6, 4
HEAD_RADIAL, HEAD_RINGS = 16, 10

# Skin weights.
JOINT_BLEND = 0.25                 # blend band across a joint, as a fraction of the
                                   # shorter adjacent segment
PART_BLEND = 3.0                   # cm over which one part's weights cross into another's.
                                   # At 5 the torso near the shoulder took enough arm weight
                                   # to drag out as a web when the arm swung.

# Preview material -- MI_Blob_Mint's teal at the spec's target shading.
SKIN_RGB = (0.06, 0.62, 0.55)
SKIN_ROUGH, SKIN_METAL = 0.18, 0.1


# =============================================================================
# Signed distance functions -- the analytic body, used for skinning after fusion
# =============================================================================

def sd_round_box(p, centre, half, r):
    q = Vector((abs(p.x - centre.x) - half.x,
                abs(p.y - centre.y) - half.y,
                abs(p.z - centre.z) - half.z))
    outside = Vector((max(q.x, 0.0), max(q.y, 0.0), max(q.z, 0.0))).length
    return outside + min(max(q.x, max(q.y, q.z)), 0.0) - r


def sd_sphere(p, centre, r):
    return (p - centre).length - r


def sd_capsule(p, a, b, r):
    ab, ap = b - a, p - a
    t = 0.0 if ab.length_squared < 1e-9 else min(1.0, max(0.0, ap.dot(ab) / ab.length_squared))
    return (ap - ab * t).length - r


def arm_segment(sy):
    """The arm capsule's axis: tucked in at the shoulder, leaning out to the wrist."""
    return (Vector((0.0, sy * ARM_TOP_Y, SHOULDER[2])),
            Vector((0.0, sy * ARM_BOT_Y, WRIST_Z)))


def body_parts():
    """The analytic body. Each entry: (name, sdf callable)."""
    torso_c = Vector((0.0, 0.0, (TORSO_Z0 + TORSO_Z1) / 2.0))
    torso_h = Vector((TORSO_D / 2.0 - TORSO_BEVEL,
                      TORSO_W / 2.0 - TORSO_BEVEL,
                      (TORSO_Z1 - TORSO_Z0) / 2.0 - TORSO_BEVEL))
    parts = [
        ("torso", lambda p: sd_round_box(p, torso_c, torso_h, TORSO_BEVEL)),
        ("head", lambda p: sd_sphere(p, Vector((0.0, 0.0, HEAD_CZ)), HEAD_D / 2.0)),
    ]
    for side, sy in (("L", -1.0), ("R", 1.0)):
        a, b = arm_segment(sy)
        parts.append(("arm" + side, lambda p, a=a, b=b: sd_capsule(p, a, b, ARM_D / 2.0)))
        c = Vector((0.0, sy * HIP[1], HIP[2]))
        d = Vector((0.0, sy * HIP[1], FOOT_Z))
        parts.append(("leg" + side, lambda p, c=c, d=d: sd_capsule(p, c, d, LEG_D / 2.0)))
    return parts


# =============================================================================
# Geometry helpers
# =============================================================================

def revolve(profile, radial):
    """Revolve a (radius, z) profile around z. A radius of 0 becomes a pole vertex,
    which is what lets this build both capsules and spheres with no cap special case."""
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
    prof = []
    for i in range(cap_rings + 1):
        a = math.pi / 2.0 * (1.0 - i / cap_rings)
        prof.append((r * math.cos(a), z_top + r * math.sin(a)))
    for i in range(1, body_rings):
        prof.append((r, z_top + (z_bot - z_top) * i / body_rings))
    for i in range(cap_rings + 1):
        a = -math.pi / 2.0 * (i / cap_rings)
        prof.append((r * math.cos(a), z_bot + r * math.sin(a)))
    return prof


def sphere_profile(cz, r, rings):
    return [(r * math.cos(math.pi / 2.0 - math.pi * i / rings),
             cz + r * math.sin(math.pi / 2.0 - math.pi * i / rings))
            for i in range(rings + 1)]


def add_geom(bm, verts, faces, offset=Vector((0, 0, 0))):
    bverts = [bm.verts.new(v + offset) for v in verts]
    for f in faces:
        try:
            bm.faces.new([bverts[i] for i in f])
        except ValueError:
            pass  # coincident face at a pole


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
    bm = bmesh.new()

    cz = (TORSO_Z0 + TORSO_Z1) / 2.0
    cube = bmesh.ops.create_cube(bm, size=1.0)["verts"]
    bmesh.ops.scale(bm, vec=Vector((TORSO_D, TORSO_W, TORSO_Z1 - TORSO_Z0)), verts=cube)
    bmesh.ops.translate(bm, vec=Vector((0.0, 0.0, cz)), verts=cube)
    bmesh.ops.bevel(bm, geom=bm.edges[:] + bm.verts[:], offset=TORSO_BEVEL,
                    segments=TORSO_BEVEL_SEGS, profile=0.5, affect="EDGES",
                    clamp_overlap=True)

    v, f = revolve(sphere_profile(HEAD_CZ, HEAD_D / 2.0, HEAD_RINGS), HEAD_RADIAL)
    add_geom(bm, v, f)

    for sy in (-1.0, 1.0):
        a, b = arm_segment(sy)
        length = (b - a).length
        v, f = revolve(capsule_profile(0.0, -length, ARM_D / 2.0,
                                       LIMB_BODY_RINGS, LIMB_CAP_RINGS), LIMB_RADIAL)
        # Built vertically then swung onto the leaning axis.
        rot = Vector((0.0, 0.0, -1.0)).rotation_difference(b - a).to_matrix()
        add_geom(bm, [rot @ p for p in v], f, a)

        v, f = revolve(capsule_profile(HIP[2], FOOT_Z, LEG_D / 2.0,
                                       LIMB_BODY_RINGS, LIMB_CAP_RINGS), LIMB_RADIAL)
        add_geom(bm, v, f, Vector((0.0, sy * HIP[1], 0.0)))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new("BlobMesh")
    bm.to_mesh(me)
    bm.free()

    obj = bpy.data.objects.new("BlobMesh", me)
    bpy.context.collection.objects.link(obj)
    return obj


def fuse(obj):
    """Voxel-remesh the intersecting primitives into one continuous skin, then
    smooth the fusion seams and decimate back to budget."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    rem = obj.modifiers.new("Remesh", "REMESH")
    rem.mode, rem.voxel_size = "VOXEL", VOXEL_SIZE
    bpy.ops.object.modifier_apply(modifier=rem.name)

    smo = obj.modifiers.new("Smooth", "SMOOTH")
    smo.factor, smo.iterations = SMOOTH_FACTOR, SMOOTH_REPEAT
    bpy.ops.object.modifier_apply(modifier=smo.name)

    tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    if tris > TARGET_TRIS:
        dec = obj.modifiers.new("Decimate", "DECIMATE")
        dec.decimate_type, dec.ratio = "COLLAPSE", TARGET_TRIS / float(tris)
        bpy.ops.object.modifier_apply(modifier=dec.name)

    # Voxel surface extraction pulls the skin slightly inside the source primitives,
    # which lifts the feet off z=0. Drop the whole mesh back onto the ground plane.
    lift = min(v.co.z for v in obj.data.vertices)
    for v in obj.data.vertices:
        v.co.z -= lift

    obj.data.shade_smooth()
    return obj


BONES = [
    # name,       head,                           tail,                          parent
    ("Pelvis",    (0, 0, PELVIS_Z),               (0, 0, PELVIS_Z + 6),          None),
    ("Spine",     (0, 0, TORSO_Z0),               (0, 0, TORSO_Z1),              "Pelvis"),
    ("Head",      (0, 0, TORSO_Z1),               (0, 0, TOTAL_H),               "Spine"),
    ("UpperArmL", (0, -SHOULDER[1], SHOULDER[2]), (0, -SHOULDER[1], ELBOW_Z),    "Spine"),
    ("ForeArmL",  (0, -SHOULDER[1], ELBOW_Z),     (0, -SHOULDER[1], WRIST_Z),    "UpperArmL"),
    ("UpperArmR", (0, SHOULDER[1], SHOULDER[2]),  (0, SHOULDER[1], ELBOW_Z),     "Spine"),
    ("ForeArmR",  (0, SHOULDER[1], ELBOW_Z),      (0, SHOULDER[1], WRIST_Z),     "UpperArmR"),
    ("ThighL",    (0, -HIP[1], HIP[2]),           (0, -HIP[1], KNEE_Z),          "Pelvis"),
    ("ShinL",     (0, -HIP[1], KNEE_Z),           (0, -HIP[1], FOOT_Z),          "ThighL"),
    ("ThighR",    (0, HIP[1], HIP[2]),            (0, HIP[1], KNEE_Z),           "Pelvis"),
    ("ShinR",     (0, HIP[1], KNEE_Z),            (0, HIP[1], FOOT_Z),           "ThighR"),
]


def build_armature():
    arm_data = bpy.data.armatures.new("BlobArmature")
    arm_obj = bpy.data.objects.new("Root", arm_data)   # -> UE root bone; see docstring
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    for name, head, tail, parent in BONES:
        b = arm_data.edit_bones.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        if parent:
            b.parent = arm_data.edit_bones[parent]
            b.use_connect = b.head == b.parent.tail
        # Explicit roll: put local X on world Y for EVERY bone, so a pitch is
        # rotation about local X regardless of which way the bone points. Blender's
        # automatic roll does not do this and it twisted the legs.
        d = (b.tail - b.head).normalized()
        b.align_roll(Vector((0.0, 1.0, 0.0)).cross(d))

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def joint_blend(z, joint_z, band):
    """1.0 = fully the upper bone, 0.0 = fully the lower one, linear across the band."""
    if band <= 1e-6:
        return 1.0 if z >= joint_z else 0.0
    return min(1.0, max(0.0, (z - (joint_z - band / 2.0)) / band))


def part_weights(name, p):
    """Bone weights a single analytic part would assign to point p."""
    if name == "torso":
        return {"Spine": 1.0}
    if name == "head":
        return {"Head": 1.0}
    side = name[-1]
    if name.startswith("arm"):
        band = JOINT_BLEND * min(SHOULDER[2] - ELBOW_Z, ELBOW_Z - WRIST_Z)
        w = joint_blend(p.z, ELBOW_Z, band)
        return {"UpperArm" + side: w, "ForeArm" + side: 1.0 - w}
    band = JOINT_BLEND * min(HIP[2] - KNEE_Z, KNEE_Z - FOOT_Z)
    w = joint_blend(p.z, KNEE_Z, band)
    return {"Thigh" + side: w, "Shin" + side: 1.0 - w}


def assign_weights(mesh_obj, arm_obj):
    """Skin the fused mesh from the analytic body.

    Bone heat is not used: after fusion there are no part boundaries left for it to
    respect, the head/torso junction would smear, and the torso is a rigid slab that
    must not deform at all. Instead every vertex is classified against the ORIGINAL
    primitives by signed distance -- which the remesh cannot disturb -- and the two
    nearest parts are cross-faded over PART_BLEND cm so the fused junctions deform
    smoothly instead of snapping at a seam.
    """
    parts = body_parts()
    groups = {name: mesh_obj.vertex_groups.new(name=name) for name, _, _, _ in BONES}

    for v in mesh_obj.data.vertices:
        p = v.co
        ds = sorted(((sdf(p), name) for name, sdf in parts), key=lambda t: t[0])
        (d1, n1), (d2, n2) = ds[0], ds[1]

        w = dict(part_weights(n1, p))
        gap = d2 - d1
        if gap < PART_BLEND:
            t = 0.5 + 0.5 * (gap / PART_BLEND)          # 0.5 at a tie -> 1.0 far apart
            w = {k: val * t for k, val in w.items()}
            for k, val in part_weights(n2, p).items():
                w[k] = w.get(k, 0.0) + val * (1.0 - t)

        total = sum(w.values()) or 1.0
        for bone, val in w.items():
            if val > 1e-4:
                groups[bone].add([v.index], val / total, "REPLACE")

    mesh_obj.parent = arm_obj
    mesh_obj.modifiers.new("Armature", "ARMATURE").object = arm_obj


def make_material():
    mat = bpy.data.materials.new("M_BlobPreview")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*SKIN_RGB, 1.0)
    b.inputs["Roughness"].default_value = SKIN_ROUGH
    b.inputs["Metallic"].default_value = SKIN_METAL
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
    scene.render.resolution_x, scene.render.resolution_y = 620, 780
    scene.world = bpy.data.worlds.new("W")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.92, .93, .93, 1)

    key = bpy.data.objects.new("Key", bpy.data.lights.new("Key", "AREA"))
    key.data.energy, key.data.size = 2.2e6, 300
    key.location = (-260, -220, 300)
    key.rotation_euler = (math.radians(48), 0, math.radians(-42))
    bpy.context.collection.objects.link(key)

    fill = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", "AREA"))
    fill.data.energy, fill.data.size = 5e5, 500
    fill.location = (300, -160, 120)
    fill.rotation_euler = (math.radians(80), 0, math.radians(62))
    bpy.context.collection.objects.link(fill)

    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    cam.data.lens = 85
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    return cam


def aim_camera(cam, yaw_deg, dist=330.0, target_z=50.0):
    """yaw 0 == facing the camera. The figure faces +X (UE actor-forward) and its
    arms spread along Y, so starting the camera on -Y sights straight down the arms."""
    a = math.radians(yaw_deg)
    cam.location = (dist * math.cos(a), dist * math.sin(a), target_z + 42.0)
    d = Vector((0, 0, target_z)) - Vector(cam.location)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def pose(arm_obj, angles):
    """angles: {bone_name: pitch_degrees}. Every bone's roll was set so local X is
    world Y, so pitch is rotation about local X for all of them."""
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

    mesh_obj = fuse(build_mesh())
    arm_obj = build_armature()
    assign_weights(mesh_obj, arm_obj)
    mesh_obj.data.materials.append(make_material())

    print("GEN_VERTS:", len(mesh_obj.data.vertices))
    print("GEN_TRIS:", sum(len(p.vertices) - 2 for p in mesh_obj.data.polygons))
    print("GEN_SHELLS:", len(mesh_obj.data.polygons))
    zs = [v.co.z for v in mesh_obj.data.vertices]
    print("GEN_HEIGHT: %.2f to %.2f" % (min(zs), max(zs)))

    cam = setup_render()
    for label, yaw in (("front", 0), ("three_quarter", 38), ("side", 90), ("back", 180)):
        aim_camera(cam, yaw)
        render_to(os.path.join(outdir, "preview_" + label))

    # Bend test. Straight limbs prove nothing about a rig.
    pose(arm_obj, {"UpperArmL": -55, "ForeArmL": -50, "UpperArmR": 38, "ForeArmR": -28,
                   "ThighL": 42, "ShinL": -60, "ThighR": -30, "ShinR": -18})
    aim_camera(cam, 34)
    render_to(os.path.join(outdir, "preview_bend"))

    # Walk-ish pose from the SIDE. A forward/back limb swing is almost invisible
    # head-on, which made the first walk render look like the pose had not applied.
    pose(arm_obj, {"UpperArmL": -28, "ForeArmL": -18, "UpperArmR": 28, "ForeArmR": -10,
                   "ThighL": 30, "ShinL": -35, "ThighR": -22, "ShinR": -8})
    aim_camera(cam, 90)
    render_to(os.path.join(outdir, "preview_walk"))
    pose(arm_obj, {})

    fbx = os.path.join(outdir, "SK_Blob.fbx")
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.export_scene.fbx(
        filepath=fbx, use_selection=True, object_types={"ARMATURE", "MESH"},
        global_scale=1.0, apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE",
        axis_forward="-Y", axis_up="Z", add_leaf_bones=False,
        use_armature_deform_only=False,   # keep Pelvis, which carries no weights
        bake_anim=False, mesh_smooth_type="FACE", path_mode="COPY")
    print("GEN_FBX:", fbx, os.path.getsize(fbx))


main()
