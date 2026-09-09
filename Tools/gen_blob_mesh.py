"""Generate SK_Blob -- the player character mesh, skeleton and skin weights.

Headless Blender. Nothing here is hand-modelled; every vertex is computed from the
constants below, so changing a proportion is a re-run, not a remodel.

    blender --background --factory-startup --python Tools/gen_blob_mesh.py -- <outdir>

Writes <outdir>/SK_Blob.fbx plus preview renders.

SHAPE STRATEGY -- read this before changing anything.

The body is SIX SEPARATE ISLANDS in one mesh: torso, head, two arms, two legs. They
are NOT welded to each other, and that is deliberate.

Earlier versions fused them into one continuous skin (first a voxel remesh, then a
metaball field). Both looked right standing still and both failed the moment a limb
moved: vertices in the arm/torso transition are weighted partly to a moving bone and
partly to a stationary one, so the surface between them stretches into a membrane.
That is what "gills" were. A single continuous skin MUST stretch there -- it is what
skinning does -- so no amount of blend-width tuning removes it, it only hides it at
small angles.

Separate islands cannot stretch. They stay looking connected because of one fact:

    A SPHERE CENTRED ON A JOINT IS INVARIANT UNDER ROTATION ABOUT THAT JOINT.

So each limb carries a ball centred exactly on its pivot -- a shoulder ball, a hip
ball -- sunk into the torso. The limb swings out of a bulge that never moves and never
separates, so there is no gap to expose and no membrane to stretch, at any angle. Each
limb island is metaball-fused WITH ITSELF (ball + capsule) for a soft shoulder, but
never with the torso.

Consequently skin weights do NO cross-part blending. Each vertex belongs wholly to one
part; the only blends are within a limb, across its own elbow or knee.

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

TORSO_Z0, TORSO_Z1 = 36.0, 71.0
TORSO_W, TORSO_D = 23.0, 16.0

# Head: a sphere floating clear of the shoulders, as in the reference.
HEAD_D = 22.0
HEAD_GAP = 3.0
HEAD_CZ = TORSO_Z1 + HEAD_GAP + HEAD_D / 2.0     # -> 85.0, crown lands on 96

# Arms. SHOULDER is the pivot and must sit ON or just inside the torso wall
# (half-width 11.5) so the shoulder ball is sunk into the body -- that is what hides
# the join. The axis then leans outward to the wrist so the arm reads as its own tube.
ARM_D = 7.5
SHOULDER = (0.0, 11.0, 66.0)
ELBOW = (0.0, 13.5, 51.0)
WRIST = (0.0, 16.0, 36.0)
SHOULDER_BALL = 5.2                # > ARM_D/2, so the shoulder reads as a deltoid bulge

# Legs. Same trick: the hip ball is centred on the hip pivot, sunk into the torso.
LEG_D = 10.0
HIP = (0.0, 7.0, 36.0)
KNEE = (0.0, 7.0, 20.0)
FOOT = (0.0, 7.0, 5.0)             # one radius up, so the capsule cap lands on z = 0
HIP_BALL = 6.0

PELVIS_Z = 36.0

# Metaball field, used WITHIN a limb only (ball + capsule -> soft shoulder), never
# across parts. Stiffness is how softly the ball melts into the limb.
MBALL_RESOLUTION = 0.55
MBALL_THRESHOLD = 0.6
MBALL_STIFFNESS = 2.6
TARGET_TRIS = 2400

TORSO_ROUND = 0.9                  # fraction of half-depth spent on corner rounding
HEAD_RADIAL, HEAD_RINGS = 20, 14

JOINT_BLEND = 0.30                 # blend band across an elbow/knee, as a fraction of
                                   # the shorter adjacent segment

SKIN_RGB = (0.06, 0.62, 0.55)
SKIN_ROUGH, SKIN_METAL = 0.18, 0.1


# =============================================================================
# Signed distance functions -- the analytic body, used to classify vertices
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


def mirror(v, sy):
    return Vector((v[0], sy * v[1], v[2]))


def body_parts():
    """The analytic body. Each entry: (name, sdf). Used only to decide which island a
    vertex belongs to, so a limb's sdf is min(its ball, its capsule)."""
    torso_c = Vector((0.0, 0.0, (TORSO_Z0 + TORSO_Z1) / 2.0))
    corner = TORSO_D / 2.0 * TORSO_ROUND
    torso_h = Vector((TORSO_D / 2.0 - corner, TORSO_W / 2.0 - corner,
                      (TORSO_Z1 - TORSO_Z0) / 2.0 - corner))
    parts = [
        ("torso", lambda p: sd_round_box(p, torso_c, torso_h, corner)),
        ("head", lambda p: sd_sphere(p, Vector((0.0, 0.0, HEAD_CZ)), HEAD_D / 2.0)),
    ]
    for side, sy in (("L", -1.0), ("R", 1.0)):
        sh, wr = mirror(SHOULDER, sy), mirror(WRIST, sy)
        parts.append(("arm" + side, lambda p, sh=sh, wr=wr: min(
            sd_sphere(p, sh, SHOULDER_BALL), sd_capsule(p, sh, wr, ARM_D / 2.0))))
        hp, ft = mirror(HIP, sy), mirror(FOOT, sy)
        parts.append(("leg" + side, lambda p, hp=hp, ft=ft: min(
            sd_sphere(p, hp, HIP_BALL), sd_capsule(p, hp, ft, LEG_D / 2.0))))
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
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target="MESH")
    return bpy.context.view_layer.objects.active


def calibrate():
    """Measure how far inside its element radius a metaball surface forms, so radii can
    be corrected automatically instead of by a hardcoded fudge."""
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


def build_limb(name, gain, joint, tip, joint_ball, limb_r):
    """One limb as its OWN metaball field: a ball on the pivot fused with the capsule.

    Its own field, not a shared one -- fusing a limb to the torso is exactly what
    produced the stretching. The distinct object name keeps Blender from blending it
    with the other limbs.
    """
    mb = bpy.data.metaballs.new(name)
    mb.resolution, mb.render_resolution = MBALL_RESOLUTION, MBALL_RESOLUTION
    mb.threshold = MBALL_THRESHOLD
    obj = bpy.data.objects.new(name, mb)
    bpy.context.collection.objects.link(obj)

    ball = mb.elements.new()
    ball.type, ball.co = "BALL", joint
    ball.radius, ball.stiffness = joint_ball / gain, MBALL_STIFFNESS

    d = tip - joint
    cap = mb.elements.new()
    cap.type, cap.co = "CAPSULE", (joint + tip) / 2.0
    cap.radius, cap.stiffness = limb_r / gain, MBALL_STIFFNESS
    cap.size_x = d.length / 2.0
    cap.rotation = Vector((1.0, 0.0, 0.0)).rotation_difference(d.normalized())

    return _to_mesh(obj)


def build_torso(gain):
    mb = bpy.data.metaballs.new("Torso")
    mb.resolution, mb.render_resolution = MBALL_RESOLUTION, MBALL_RESOLUTION
    mb.threshold = MBALL_THRESHOLD
    obj = bpy.data.objects.new("Torso", mb)
    bpy.context.collection.objects.link(obj)

    round_r = TORSO_D / 2.0 * TORSO_ROUND
    e = mb.elements.new()
    e.type, e.co = "CUBE", (0.0, 0.0, (TORSO_Z0 + TORSO_Z1) / 2.0)
    e.radius, e.stiffness = round_r / gain, MBALL_STIFFNESS
    e.size_x = max(TORSO_D / 2.0 - round_r, 0.05)
    e.size_y = max(TORSO_W / 2.0 - round_r, 0.05)
    e.size_z = max((TORSO_Z1 - TORSO_Z0) / 2.0 - round_r, 0.05)
    return _to_mesh(obj)


def build_head():
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


def assemble(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "BlobMesh"

    tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    if tris > TARGET_TRIS:
        dec = obj.modifiers.new("Decimate", "DECIMATE")
        dec.decimate_type, dec.ratio = "COLLAPSE", TARGET_TRIS / float(tris)
        bpy.ops.object.modifier_apply(modifier=dec.name)

    # Metaball surfaces form inside their elements, lifting the feet off the floor.
    lift = min(v.co.z for v in obj.data.vertices)
    for v in obj.data.vertices:
        v.co.z -= lift

    obj.data.shade_smooth()
    return obj


def bone_table():
    b = [("Pelvis", (0, 0, PELVIS_Z), (0, 0, PELVIS_Z + 6), None),
         ("Spine", (0, 0, TORSO_Z0), (0, 0, TORSO_Z1), "Pelvis"),
         ("Head", (0, 0, TORSO_Z1), (0, 0, TOTAL_H), "Spine")]
    for side, sy in (("L", -1.0), ("R", 1.0)):
        sh, el, wr = mirror(SHOULDER, sy), mirror(ELBOW, sy), mirror(WRIST, sy)
        b.append(("UpperArm" + side, sh, el, "Spine"))
        b.append(("ForeArm" + side, el, wr, "UpperArm" + side))
        hp, kn, ft = mirror(HIP, sy), mirror(KNEE, sy), mirror(FOOT, sy)
        b.append(("Thigh" + side, hp, kn, "Pelvis"))
        b.append(("Shin" + side, kn, ft, "Thigh" + side))
    return b


BONES = bone_table()


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
        # Explicit roll: local X on world Y for EVERY bone, so pitch is rotation about
        # local X whichever way the bone points. Blender's automatic roll twisted the
        # legs.
        d = (b.tail - b.head).normalized()
        b.align_roll(Vector((0.0, 1.0, 0.0)).cross(d))

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def along(p, a, b):
    """Fraction of the way from a to b, projected onto the segment."""
    ab, ap = b - a, p - a
    if ab.length_squared < 1e-9:
        return 0.0
    return min(1.0, max(0.0, ap.dot(ab) / ab.length_squared))


def part_weights(name, p, sy_lookup):
    """Weights for a vertex known to belong to `name`.

    No cross-part term: islands never share vertices, so the only blend is a limb's
    own elbow or knee. Measured along the limb axis rather than by z, because the arm
    axis leans outward.
    """
    if name == "torso":
        return {"Spine": 1.0}
    if name == "head":
        return {"Head": 1.0}

    side = name[-1]
    sy = sy_lookup[side]
    if name.startswith("arm"):
        joint, mid, tip = mirror(SHOULDER, sy), mirror(ELBOW, sy), mirror(WRIST, sy)
        upper, lower = "UpperArm" + side, "ForeArm" + side
    else:
        joint, mid, tip = mirror(HIP, sy), mirror(KNEE, sy), mirror(FOOT, sy)
        upper, lower = "Thigh" + side, "Shin" + side

    t = along(p, joint, tip)
    t_mid = along(mid, joint, tip)
    band = JOINT_BLEND * min(t_mid, 1.0 - t_mid)
    if band <= 1e-6:
        w = 1.0 if t <= t_mid else 0.0
    else:
        w = 1.0 - min(1.0, max(0.0, (t - (t_mid - band / 2.0)) / band))
    return {upper: w, lower: 1.0 - w}


def weight_island(obj, part_name, sy):
    """Weight ONE island, before anything is joined.

    Weights are assigned per island rather than by classifying vertices against the
    analytic body, because the two disagree exactly where it matters: the shoulder and
    hip balls are deliberately sunk INSIDE the torso, so torso vertices around them are
    nearer the limb's surface than the torso's. Classifying by distance handed those
    vertices to the limb, and they tore off with it when it swung. The island already
    knows what it is -- use that.

    Every island declares every group so that join() merges them by name consistently.
    """
    groups = {name: obj.vertex_groups.new(name=name) for name, _, _, _ in BONES}
    for v in obj.data.vertices:
        w = part_weights(part_name, v.co, {"L": -1.0, "R": 1.0}) if sy is None else \
            part_weights(part_name, v.co, {part_name[-1]: sy})
        total = sum(w.values()) or 1.0
        for bone, val in w.items():
            if val > 1e-4:
                groups[bone].add([v.index], val / total, "REPLACE")


def bind(mesh_obj, arm_obj):
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
    """yaw 0 == facing the camera. The figure faces +X and its arms spread along Y, so
    starting the camera on -Y sights straight down the arms."""
    a = math.radians(yaw_deg)
    cam.location = (dist * math.cos(a), dist * math.sin(a), target_z + 42.0)
    d = Vector((0, 0, target_z)) - Vector(cam.location)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def pose(arm_obj, angles):
    """angles: {bone_name: pitch_degrees}. Every bone's roll puts local X on world Y,
    so pitch is rotation about local X for all of them."""
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

    # (object, part name, side sign). Each island is weighted from its own identity
    # before any joining -- see weight_island().
    islands = [(build_torso(gain), "torso", None), (build_head(), "head", None)]
    for side, sy in (("L", -1.0), ("R", 1.0)):
        islands.append((build_limb("Arm" + side, gain, mirror(SHOULDER, sy),
                                   mirror(WRIST, sy), SHOULDER_BALL, ARM_D / 2.0),
                        "arm" + side, sy))
        islands.append((build_limb("Leg" + side, gain, mirror(HIP, sy),
                                   mirror(FOOT, sy), HIP_BALL, LEG_D / 2.0),
                        "leg" + side, sy))

    for obj, part_name, sy in islands:
        weight_island(obj, part_name, sy)

    mesh_obj = assemble([o for o, _, _ in islands])
    arm_obj = build_armature()
    bind(mesh_obj, arm_obj)
    mesh_obj.data.materials.append(make_material())

    zs = [v.co.z for v in mesh_obj.data.vertices]
    print("GEN_VERTS:", len(mesh_obj.data.vertices))
    print("GEN_TRIS:", sum(len(p.vertices) - 2 for p in mesh_obj.data.polygons))
    print("GEN_ISLANDS:", len(islands))
    print("GEN_HEIGHT: %.2f to %.2f" % (min(zs), max(zs)))

    cam = setup_render()
    for label, yaw in (("front", 0), ("three_quarter", 38), ("side", 90), ("back", 180)):
        aim_camera(cam, yaw)
        render_to(os.path.join(outdir, "preview_" + label))

    # Extreme angles. If a join is going to tear or stretch, it shows here.
    pose(arm_obj, {"UpperArmL": -55, "ForeArmL": -50, "UpperArmR": 38, "ForeArmR": -28,
                   "ThighL": 42, "ShinL": -60, "ThighR": -30, "ShinR": -18})
    aim_camera(cam, 34)
    render_to(os.path.join(outdir, "preview_bend"))

    # Walk pose from the SIDE. A forward/back swing is almost invisible head-on.
    pose(arm_obj, {"UpperArmL": -28, "ForeArmL": -18, "UpperArmR": 28, "ForeArmR": -10,
                   "ThighL": 30, "ShinL": -35, "ThighR": -22, "ShinR": -8})
    aim_camera(cam, 90)
    render_to(os.path.join(outdir, "preview_walk"))

    # Same walk pose, three-quarter: the shoulder join is most exposed from here.
    aim_camera(cam, 38)
    render_to(os.path.join(outdir, "preview_walk_34"))
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
