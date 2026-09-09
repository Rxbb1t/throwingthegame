"""Generate SK_Blob -- the player character mesh, skeleton and skin weights.

Headless Blender. Nothing here is hand-modelled; every vertex is computed from the
constants below, so changing a proportion is a re-run, not a remodel.

    blender --background --factory-startup --python Tools/gen_blob_mesh.py -- <outdir>

Writes <outdir>/SK_Blob.fbx plus preview renders.

SHAPE STRATEGY -- read this before changing anything.

The reference (Snaptic) is a DETACHED sphere head above a single continuous
torso-arms-legs form with soft, blobby shoulders and hips. Two separate problems:

  * The body must be a SMOOTH union. An earlier version voxel-remeshed overlapping
    primitives, but a remesh is a HARD union: it leaves a sharp crease where the arm
    meets the torso, which read as gills. The body is therefore a METABALL field --
    Blender metaballs are a smooth (blobby) union natively, which is exactly the
    reference's shoulder treatment.
  * The head must NOT participate in that union, or it welds to the torso. It is
    built as an ordinary sphere mesh and joined afterwards as a separate island.

Metaball surfaces form where the summed field crosses `threshold`, so they sit some
way INSIDE the element radii. Rather than hand-tuning that, calibrate() measures the
shrink on a test ball and every radius is divided through by it. Change the threshold
or stiffness and the calibration follows automatically.

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

TOTAL_H = 96.0

# Torso: a rounded slab.
TORSO_Z0, TORSO_Z1 = 36.0, 71.0
TORSO_W, TORSO_D = 23.0, 16.0

# Head: a sphere floating clear of the shoulders, as in the reference. HEAD_GAP is
# the air between torso top and head underside -- the thing that makes it read as a
# separate ball rather than a lollipop.
HEAD_D = 22.0
HEAD_GAP = 3.0
HEAD_CZ = TORSO_Z1 + HEAD_GAP + HEAD_D / 2.0     # -> 85.0, so the crown lands on 96

# Arms. Bone chain shoulder -> elbow -> wrist, vertical. The metaball capsule leans
# outward so it merges into the torso at the shoulder and swings clear below it.
ARM_D = 7.5
SHOULDER = (0.0, 15.0, 66.0)
ARM_TOP_Y, ARM_BOT_Y = 13.5, 18.0
ELBOW_Z = 51.0
WRIST_Z = 36.0

# Legs. The foot point sits one radius above the ground so the capsule's bottom cap
# lands on z = 0 rather than 5 cm underneath it.
LEG_D = 10.0
HIP = (0.0, 7.0, 36.0)
KNEE_Z = 20.0
FOOT_Z = 5.0

PELVIS_Z = 36.0

# Metaball field. STIFFNESS controls how eagerly neighbouring elements blend into one
# another: higher is blobbier and softer, lower keeps limbs distinct. This is the knob
# for "gills vs melted".
MBALL_RESOLUTION = 0.7
MBALL_THRESHOLD = 0.6
MBALL_STIFFNESS = 2.6
TARGET_TRIS = 2200

HEAD_RADIAL, HEAD_RINGS = 20, 14

# Skin weights.
JOINT_BLEND = 0.25                 # blend band across a joint, as a fraction of the
                                   # shorter adjacent segment
PART_BLEND = 3.0                   # cm over which one part's weights cross into another's

# Preview material -- MI_Blob_Mint's teal at the spec's target shading.
SKIN_RGB = (0.06, 0.62, 0.55)
SKIN_ROUGH, SKIN_METAL = 0.18, 0.1


# =============================================================================
# Signed distance functions -- the analytic body, used for skinning after meshing
# =============================================================================

def sd_round_box(p, centre, half, r):
    q = Vector((abs(p.x - centre.x) - half.x,
                abs(p.y - centre.y) - half.y,
                abs(p.z - centre.z) - half.z))
    return (Vector((max(q.x, 0.0), max(q.y, 0.0), max(q.z, 0.0))).length
            + min(max(q.x, max(q.y, q.z)), 0.0) - r)


def sd_sphere(p, centre, r):
    return (p - centre).length - r


def sd_capsule(p, a, b, r):
    ab, ap = b - a, p - a
    t = 0.0 if ab.length_squared < 1e-9 else min(1.0, max(0.0, ap.dot(ab) / ab.length_squared))
    return (ap - ab * t).length - r


def arm_segment(sy):
    """Arm axis: tucked into the torso at the shoulder, leaning out to the wrist."""
    return (Vector((0.0, sy * ARM_TOP_Y, SHOULDER[2])),
            Vector((0.0, sy * ARM_BOT_Y, WRIST_Z)))


def leg_segment(sy):
    return (Vector((0.0, sy * HIP[1], HIP[2])), Vector((0.0, sy * HIP[1], FOOT_Z)))


def body_parts():
    """The analytic body. Each entry: (name, sdf callable)."""
    torso_c = Vector((0.0, 0.0, (TORSO_Z0 + TORSO_Z1) / 2.0))
    corner = TORSO_D / 2.0 * 0.9
    torso_h = Vector((TORSO_D / 2.0 - corner, TORSO_W / 2.0 - corner,
                      (TORSO_Z1 - TORSO_Z0) / 2.0 - corner))
    parts = [
        ("torso", lambda p: sd_round_box(p, torso_c, torso_h, corner)),
        ("head", lambda p: sd_sphere(p, Vector((0.0, 0.0, HEAD_CZ)), HEAD_D / 2.0)),
    ]
    for side, sy in (("L", -1.0), ("R", 1.0)):
        a, b = arm_segment(sy)
        parts.append(("arm" + side, lambda p, a=a, b=b: sd_capsule(p, a, b, ARM_D / 2.0)))
        c, d = leg_segment(sy)
        parts.append(("leg" + side, lambda p, c=c, d=d: sd_capsule(p, c, d, LEG_D / 2.0)))
    return parts


# =============================================================================
# Build
# =============================================================================

def wipe():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.metaballs, bpy.data.armatures,
                  bpy.data.objects, bpy.data.materials, bpy.data.cameras,
                  bpy.data.lights):
        for item in list(block):
            block.remove(item)


def _to_mesh(obj):
    """Convert a metaball object to a mesh and return the resulting object."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target="MESH")
    return bpy.context.view_layer.objects.active


def calibrate():
    """Measure how far inside its element radius a metaball surface actually forms.

    Returns the factor to divide radii by. Doing this rather than hard-coding a fudge
    means threshold and stiffness stay free to tune without breaking every dimension.
    """
    mb = bpy.data.metaballs.new("Cal")
    mb.resolution, mb.threshold = 0.25, MBALL_THRESHOLD
    obj = bpy.data.objects.new("Cal", mb)
    bpy.context.collection.objects.link(obj)
    e = mb.elements.new()
    e.type, e.co, e.radius, e.stiffness = "BALL", (0, 0, 0), 10.0, MBALL_STIFFNESS

    mesh_obj = _to_mesh(obj)
    actual = max(v.co.length for v in mesh_obj.data.vertices)
    bpy.data.objects.remove(mesh_obj, do_unlink=True)
    return actual / 10.0


def build_body(gain):
    """Torso + arms + legs as one smooth metaball union. No head."""
    mb = bpy.data.metaballs.new("Body")
    mb.resolution, mb.render_resolution = MBALL_RESOLUTION, MBALL_RESOLUTION
    mb.threshold = MBALL_THRESHOLD
    obj = bpy.data.objects.new("Body", mb)
    bpy.context.collection.objects.link(obj)

    def elem(kind, co, radius, size=None, direction=None):
        e = mb.elements.new()
        e.type, e.co = kind, co
        e.radius, e.stiffness = radius / gain, MBALL_STIFFNESS
        if size:
            e.size_x, e.size_y, e.size_z = size
        if direction:
            e.rotation = Vector((1.0, 0.0, 0.0)).rotation_difference(direction)
        return e

    # Torso. A CUBE element is a rounded box: size_* is the flat core, radius the
    # rounding around it, so the two together make the slab.
    round_r = TORSO_D / 2.0 * 0.9
    core = Vector((max(TORSO_D / 2.0 - round_r, 0.1),
                   max(TORSO_W / 2.0 - round_r, 0.1),
                   max((TORSO_Z1 - TORSO_Z0) / 2.0 - round_r, 0.1)))
    elem("CUBE", (0.0, 0.0, (TORSO_Z0 + TORSO_Z1) / 2.0), round_r, size=core)

    for sy in (-1.0, 1.0):
        for (a, b), r in ((arm_segment(sy), ARM_D / 2.0), (leg_segment(sy), LEG_D / 2.0)):
            d = b - a
            elem("CAPSULE", (a + b) / 2.0, r,
                 size=(d.length / 2.0, 0.0, 0.0), direction=d.normalized())

    return _to_mesh(obj)


def build_head():
    """The head is an ordinary sphere, deliberately OUTSIDE the metaball field: put it
    in the field and it welds to the shoulders, and the reference's head is clearly a
    separate ball."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=HEAD_RADIAL, v_segments=HEAD_RINGS,
                              radius=HEAD_D / 2.0)
    bmesh.ops.translate(bm, vec=Vector((0.0, 0.0, HEAD_CZ)), verts=bm.verts[:])
    me = bpy.data.meshes.new("Head")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("Head", me)
    bpy.context.collection.objects.link(obj)
    return obj


def assemble(body, head):
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    head.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "BlobMesh"

    tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    if tris > TARGET_TRIS:
        dec = obj.modifiers.new("Decimate", "DECIMATE")
        dec.decimate_type, dec.ratio = "COLLAPSE", TARGET_TRIS / float(tris)
        bpy.ops.object.modifier_apply(modifier=dec.name)

    # The metaball surface forms inside its elements, which lifts the feet off the
    # floor. Drop the whole figure back onto z = 0.
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
        # Explicit roll: put local X on world Y for EVERY bone, so pitch is rotation
        # about local X whichever way the bone points. Blender's automatic roll does
        # not do this and it twisted the legs.
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
    """Skin from the analytic body rather than by bone heat.

    Metaball meshing produces topology with no relationship to the parts that made it,
    and the torso is a slab that must not deform. So every vertex is classified by
    signed distance to the ORIGINAL primitives and the two nearest parts cross-fade
    over PART_BLEND cm. Head and torso are excluded from that cross-fade -- they are
    separate islands with air between them, and blending would drag the torso's top
    around whenever the head moved.
    """
    parts = body_parts()
    groups = {name: mesh_obj.vertex_groups.new(name=name) for name, _, _, _ in BONES}

    for v in mesh_obj.data.vertices:
        p = v.co
        ds = sorted(((sdf(p), name) for name, sdf in parts), key=lambda t: t[0])
        (d1, n1), (d2, n2) = ds[0], ds[1]

        w = dict(part_weights(n1, p))
        detached = {n1, n2} == {"head", "torso"}
        if not detached and d2 - d1 < PART_BLEND:
            t = 0.5 + 0.5 * ((d2 - d1) / PART_BLEND)
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
    """yaw 0 == facing the camera. The figure faces +X (UE actor-forward) and its arms
    spread along Y, so starting the camera on -Y sights straight down the arms."""
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

    gain = calibrate()
    print("GEN_MBALL_GAIN: %.4f" % gain)

    mesh_obj = assemble(build_body(gain), build_head())
    arm_obj = build_armature()
    assign_weights(mesh_obj, arm_obj)
    mesh_obj.data.materials.append(make_material())

    zs = [v.co.z for v in mesh_obj.data.vertices]
    ys = [abs(v.co.y) for v in mesh_obj.data.vertices]
    print("GEN_VERTS:", len(mesh_obj.data.vertices))
    print("GEN_TRIS:", sum(len(p.vertices) - 2 for p in mesh_obj.data.polygons))
    print("GEN_HEIGHT: %.2f to %.2f" % (min(zs), max(zs)))
    print("GEN_HALFWIDTH: %.2f" % max(ys))

    cam = setup_render()
    for label, yaw in (("front", 0), ("three_quarter", 38), ("side", 90), ("back", 180)):
        aim_camera(cam, yaw)
        render_to(os.path.join(outdir, "preview_" + label))

    pose(arm_obj, {"UpperArmL": -55, "ForeArmL": -50, "UpperArmR": 38, "ForeArmR": -28,
                   "ThighL": 42, "ShinL": -60, "ThighR": -30, "ShinR": -18})
    aim_camera(cam, 34)
    render_to(os.path.join(outdir, "preview_bend"))

    # Walk pose from the SIDE. A forward/back limb swing is almost invisible head-on,
    # which made an earlier walk render look like the pose had not applied at all.
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
