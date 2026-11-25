---
sitemap: false
layout: project
item_id: 6
name: AI-based Surgical Planning Pipeline
title: AI-based Surgical Planning Pipeline
tags: ["Python", "PyTorch", "MONAI", "3D Slicer", "FastAPI", "React", "Medical Imaging"]
summary: "A pipeline that turns a raw CT angiography scan into a complete plan for EVAR: Deep Learning Segmentation, automatic measurements that lead to stent-graft selection and C-arm angles recommendations that can be reviewed and edited by surgeons in a web app."
venue: "Medical Image Analysis, Decision Support System"
course_name: "Ph.D. work · University of Trento & S. Chiara Hospital, Trento"
completition_date: "2023 - 2026"
links:
  paper: "https://doi.org/10.1038/s41598-025-00484-7"
---

One of the most complicated project I've done: taking an input a raw CT angiography (CTA) scan of a patient with an abdominal aortic aneurysm and producing a complete plan for **Endovascular Aneurysm Repair (EVAR)** that returns: 3D segmentations of the aorta and its branches, every measurement in the European Society for Vascular Surgery (ESVS) planning guidelines, the stent-graft combination to use, and the C-arm angles suggestion for deployment.

Basically, in one sentence, a processing backend runs in on a standard hospital workstation; surgeons then review the output which can be adjust, then they can approve the plan in a web app instead of measuring everything by hand.

<figure>
  <img src="{{ '/projects/images/surgical-planning-pipeline.jpg' | relative_url }}" alt="The five steps of the segmentation pipeline: initial segmentation with nnU-Net, vessel identification, thrombus segmentation with SegResNet, areas localization and features extraction">
  <figcaption>The pipeline visually explained, from the raw CTA to the extracted features.</figcaption>
</figure>

## What has been used

| Layer | Tools |
| --- | --- |
| Language | Python 3 (mainly), Javascript |
| Imaging and geometry | 3D Slicer (scripted Python API), VTK, VMTK, SimpleITK, scikit-image |
| Deep learning | PyTorch, MONAI, TotalSegmentator |
| Standard ML | scikit-learn |
| Training |  Weights & Biases|
| Web BackEnd | FastAPI |
| Web FrontEnd | React |


## Architecture

The feature-extraction pipeline is composed in seven stages, each is inside 3D Slicer (a lot of things were already implemented in Slicer, such as centerline extraction, which came very in handy! it saved me a lot of time...):

```bash
Slicer --python-script step5_centerline_cpr.py <case_id>
```

Stages share no in-memory state. Each one loads what the previous stage persisted and saves its own output:

```text
<case>/Steps/
  step5.mrb          # full Slicer scene: volumes, segmentations, transforms
  step5_state.json   # thresholds, bounds, landmark indices, features
  arrays/*.npy       # arrays that cannot live in the scene
```

This makes the pipeline "**recoverable**" (say that a certain CTA raises an error, we can restart the pipeline from the last stage, not from scratch), **debuggable** (any stage can be rerun alone while tuning it), and **reviewable**: a clinician can open any intermediate scene in Slicer, fix a segmentation (if the model fails at doing so), save, and let the next stage continue from there.

| Stage | Functionality 
| --- | --- 
| 1. Preparation | Load the CT and TotalSegmentator masks, fit per-patient intensity models for adaptive normalisation 
| 2. Localization | Extract bright vascular structures (like the renal arteries) which sits around the aorta (within certain mms)
| 3. Candidate vessels | Split them into connected components
| 4. Vessel labelling | Label the branches and extract their ostium  (where they bifurcate basically)
| 5. Centerlines & CPR | Extract centerlines and straighten both aorto-iliac axes 
| 6. Features | Thrombus segmentation, landmarks, diameters, angles, sac and wall analysis 
| 7. Export | Lumen tree, NIfTI masks, radiomics, final scene

## Some implementation highlights

### Adaptive intensity normalisation

CTAs from different hospitals use different scanners, different contrast timing, different evrythin; so fixed Hounsfield-unit (HU) thresholds break. In stage one we fit a small intensity model for each patient and each vessel (aorta, left iliac, right iliac, because contrast medium in the left iliac is "less strong" than in the aorta!) instead:

```python
def fit_intensity_model(volume, vessel_masks):
    hu = volume[np.logical_or.reduce(vessel_masks)].reshape(-1, 1)
    hu = hu[IsolationForest(contamination=0.05).fit_predict(hu) == 1]  # drop outliers
    gmm = GaussianMixture(n_components=2).fit(hu)
    lo, hi = np.sort(gmm.means_.ravel())  # non-enhanced tissue, contrast-filled blood

    counts, edges = np.histogram(hu, bins=100)
    centers = (edges[:-1] + edges[1:]) / 2
    between = (centers > lo) & (centers < hi)
    threshold = centers[between][np.argmin(counts[between])]  # histogram valley
    return gmm, threshold
```

These statistics drive everything downstream: we use them to distinct lumen versus thrombus, locating calcification, and to normalise the input for a neural network later in the pipeline.

### Working in "straightened" space

The VMTK library extracts a centerline from the aorta above the kidneys into each common iliac artery. Then, using Slicer's curved planar reformation module we resample the CT along it: a 64 × 64 mm cross-section every millimeter, always perpendicular to the vessel. Segmentations are moved into this space with a transformation and back with its inverse.

Whta is nice about this is that most measurements become simple per-slice NumPy operations. For instance, the aortic bifurcation is the last slice before the lumen splits into two connected components:

```python
n_parts = []
for z in range(z_start, z_end):  # walk down the straightened lumen
    parts = [p for p in regionprops(label(lumen[z])) if p.area > min_area]
    n_parts.append(len(parts))
n_parts = median_filter(np.array(n_parts), size=3)  # ignore single noisy slices

split = int(np.argmax(n_parts >= 2))
bifurcation = z_start + max(z for z in range(split + 1) if n_parts[z] <= 1)
```

Much easier.

### Measuring diameters: three ways

Diameters are the most important measurements in this project: they decide whether the neck is suitable and which stent-graft sizes fit. An aneurysmal aorta is rarely circular, though, and a naive measurement on an axial CT slice overestimates it wherever the vessel is tilted, because an oblique cut through a tube is an ellipse. We therefore measures every cross-section **perpendicular to the centerline**, in three ways:

**1. Minimum and maximum Feret diameters on the straightened volume:** In straightened space every slice is already perpendicular to the vessel, so this is a 2D problem on each mask slice, we can just use the feret module to extract the Min and Max feret diameter:

```python
from feret.main import Calculater

def feret_diameters(mask_slice, spacing_mm):
    calc = Calculater(mask_slice, edge=False)
    calc.calculate_minferet()
    calc.calculate_maxferet()
    return {
        "min_mm": calc.minf * spacing_mm, "min_angle": calc.minf_angle,
        "max_mm": calc.maxf * spacing_mm, "max_coords": calc.maxf_coords,
    }
```

Each measurement is stored with its slice index, angle and endpoints, not just its value. These are exactly the red and yellow segments that the web app draws back on the slice (see image below!), so every number in the plan can be traced to the pixels it came from, which will become very important later on!

**2. Area-equivalent (CE) diameter from the 3D surface:** On the original, non-straightened geometry, the pipeline cuts the closed 3D surface of the segmentation with a plane at each point of the centerline extracted, Slicer gives a full coordinate frame for every point of the curve: its position, plus a z-axis tangent to the vessel that serves as the plane normal.

```python
def area_equivalent_diameter(centerline, i, surface):
    frame = vtk.vtkMatrix4x4()
    centerline.GetCurvePointToWorldTransformAtPointIndex(i, frame)
    center = [frame.GetElement(r, 3) for r in range(3)]  # point on the centerline
    normal = [frame.GetElement(r, 2) for r in range(3)]  # tangent to the vessel

    plane = vtk.vtkPlane()
    plane.SetOrigin(center)
    plane.SetNormal(normal)

    cutter = vtk.vtkCutter()                 # surface ∩ plane -> contour lines
    cutter.SetInputData(surface)
    cutter.SetCutFunction(plane)

    # keep only the loop around this centerline point (drops cuts through other vessels)
    loop = vtk.vtkConnectivityFilter()
    loop.SetInputConnection(cutter.GetOutputPort())
    loop.SetExtractionModeToClosestPointRegion()
    loop.SetClosestPoint(center)

    fill = vtk.vtkContourTriangulator()      # closed contour -> filled polygon
    fill.SetInputConnection(loop.GetOutputPort())

    props = vtk.vtkMassProperties()
    props.SetInputConnection(fill.GetOutputPort())
    props.Update()
    return 2 * np.sqrt(props.GetSurfaceArea() / np.pi)  # circle with the same area
```

The connectivity step matters near the bifurcation and the renal arteries. There, the plane also slices through neighbouring vessels, and only the contour closest to the centerline point belongs to the aorta. Because the diameter comes from the area, a single irregular bulge cannot dominate it the way it can dominate a caliper measurement. The same function also produces the landing-zone contours shown in the web app's 3D C-arm preview (image below!).

**3. Maximum inscribed sphere:** VMTK stores, for each point in the centerline, the radius of the largest sphere that fits inside the egmentation. This gives a third, conservative estimate at no extra cost, interpolated onto the resampled centerline.

The result is a **diameter profile along the vessel**. We can find where the proximal neck ends by looking for where the profile grows beyond a set fraction of its value at the lowest renal artery. Then we can report neck diameters at 0, 5, 10, 15 and 20 mms below the lowest renal artery, each averaged over neighbouring centerline points to reduce the variance in our estimation.

### Thrombus segmentation

TotalSegmentator is very reliable on healthy anatomy but not on the **diseased aorta**. In an aneurysm, the contrast-filled lumen is surrounded by thrombus, which has almost the same intensity as the tissue around the aorta (for instance, the duodenum), so the outer wall is hard to see, even for a surgeon! We noticed that TotalSegmentator tends to stop at the lumen frequently: our hypothesis is that its training data is dominated by aortas whose visible boundary is just the lumen (healthy vessels, or scans with a different contrast phase), so it learns to stop at the first strong contrast edge it meets, but I guess we'll never know for sure. What is certain though is that for EVAR planning that is a serious error: the stent-graft is sized on the outer diameter, and the lumen underestimates it.

<div class="project-media-row">
  <figure>
    <img src="{{ '/projects/images/surgical-planning-ct-aneurysm.jpg' | relative_url }}" alt="Axial CT angiography slice through an abdominal aortic aneurysm: a bright contrast-filled lumen surrounded by a grey thrombus that blends with the surrounding tissue">
    <figcaption>An axial CTA slice through the aneurysm. The bright lumen is easy to see; the thrombus around it, and therefore the outer wall, is not.</figcaption>
  </figure>
  <figure>
    <img src="{{ '/projects/images/surgical-planning-ts-vs-segresnet.jpg' | relative_url }}" alt="The same slice with two contours: a yellow contour following only the bright lumen, and a larger red contour enclosing lumen and thrombus">
    <figcaption><span style="color:#c9a800;font-weight:600">Yellow</span>: TotalSegmentator, which follows only the lumen. <span style="color:#c0392b;font-weight:600">Red</span>: our SegResNet with the adaptive window, which includes the thrombus, which is not perfect, but a much closer estimate of the true aortic wall.</figcaption>
  </figure>
</div>

We tried to find a sufficiently large annotated dataset of EVAR patients online, but it is more probable to win the lottery... in fact, this was one of the greatest challenges of this work. To overcome this limitation, we tried to leverage the output of a TotalSegmentator model to make the task easuer for a smaller, specialized model.

- **Model**: MONAI `SegResNet` (32 initial filters, encoder blocks `[1, 2, 2, 4]`, dropout 0.2, 2 classes), in a 2D slice-by-slice and a 3D variant whose predictions are combined.
- **Training**: 200 annotated CTA scans, Dice + cross-entropy loss, Adam with polynomial learning-rate decay, random flips, 90° rotations and small affine transforms, tracked in Weights & Biases.
- **Geometric normalization**: inputs are the straightened 64 × 64 mm cross-sections, so the vessel is always centred, with the same orientation and scale, pretty convenient!
- **Intensity normalization**: an adaptive window built from the patient's own intensity model, fitted inside the TotalSegmentator masks, which are not precise enough for measuring but more than good enough for statistics!

```python
def adaptive_window(x, mu, sigma):
    # mu, sigma: this patient's non-enhanced tissue component (from the GMM)
    lo, hi = mu - 2 * sigma, mu + 2 * sigma
    return np.clip((x - lo) / (hi - lo), 0, 1)

preprocess = Compose([
    ResizeWithPadOrCropd(keys="image", spatial_size=(64, 64, -1)),
    DivisiblePadd(keys="image", k=(-1, -1, 8)),
    MedianSmoothd(keys="image", radius=1),
    ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
])
```

Contrast-filled blood saturates to bright (1 in this case), fat to dark (-1), and the thrombus fills the whole input range, whatever the scanner. Together with the fixed geometry, this works as a lightweight per-patient **domain adaptation**: every new scan is mapped into the domain the small model was trained on, this makes it possible to have very good performance even with a restricted training set (at the cost of being reliant on TotalSegmentator for the statistics!). Padding is undone with MONAI's invertible transforms, and the mask is mapped back to the original CT with the inverse straightening transform. The model reaches a Dice score of **0.96** on held-out scans.

<figure>
  <img src="{{ '/projects/images/surgical-planning-anatomy.jpg' | relative_url }}" alt="3D reconstructions of four patients with the aneurysm, sealing zones, calcifications, thrombus and secondary vessels shown in different colours">
  <figcaption>The output of the Pipeline for four patients: sealing zones, aneurysm lumen, thrombus, calcifications and secondary vessels.</figcaption>
</figure>

### Stent-graft selection

The measurements extracted are matched against the instructions for use (IFUs) of two manufacturers (Medtronic and GORE, the main manufacturers used by vascular surgeons basically) to pick the main body and iliac limbs that maximize sealing-zone coverage. The tool also reports the expected overlap between components and the residual distance to the ideal length, so the surgeon knows how much margin the plan has if the anatomy shifts during the procedure (see figure below!).

### C-arm angulation

A good projection chosen before the procedure means fewer test acquisitions in the operating room: less contrast agent for the patient, which can harm the kidneys, and less radiation for the surgical team, and less is good! To deploy the stent-graft right below the lowest renal artery (or right above an internal iliac artery), the X-ray view must show the branch's **ostium in profile** and the **aortic cross-section edge-on**, so parallax does not shift the landing zone. Both conditions hold for a viewing direction perpendicular to the branch direction *b* and to the cross-section normal *n*, which is their cross product. However, real C-arms cannot reach every angle: especially for the iliac arteries, the cross product is often a direction that the C-Arm cannot reach, so the tool searches the reachable angles for the one closest to it. A simplified sketch:

```python
b = unit(ostium_direction)            # out of the aorta, into the branch
n = unit(centerline_tangent)          # normal of the aortic cross-section at the ostium
v = unit(np.cross(n, b))              # ideal viewing direction

def loss(lao, cra):                   # negative LAO = RAO, negative CRA = CAU
    d = view_direction(lao, cra)      # C-arm angles -> X-ray beam direction
    deviation = np.degrees(np.arccos(abs(d @ v)))
    return deviation + limits_penalty(lao, cra)  # e.g. CAU > 35°, RAO > 45°

# since the possible angles are constant and they are not so many, we can score them all and pick the best one!
angles = [(lao, cra) for lao in range(-90, 91) for cra in range(-90, 91)]
best_lao, best_cra = min(angles, key=lambda a: loss(*a))
```

## Visualizing and Editing the Measurements with a Web App

We thought that by filling the ESVS EVAR guideline sheet with the measurements the surgeons would be happy, we were wrong.. 

We ran a controlled study with junior surgeons: they received an EVAR planning sheet presented as produced by AI, but the measurements actually came from a **senior vascular surgeon**! Some trainees still rejected the sheet and re-did the plan themselves, because they did not trust an algorithm. This is a clear case of **algorithmic aversion**: the label "AI" lowered trust in values that were in fact expert-made.

So we developed an interface in order to improve transparency and put the surgeon in control rather than providing him with a final answer to accept or reject. In details:

- **Every measurement is traceable:** Each value in the planning sheet is a button that opens the place where it was measured. The cross-section with the maximum (red) and minimum (yellow) diameters drawn on it, or the path along which a length was taken.
- **Every measurement is editable:** Surgeons can scroll through the straightened vessel in axial, coronal and sagittal views, adjust the window, rotate the vessel and drag the diameter endpoints if he is not satisfied with the measurement.
- **The C-arm view is previewed in 3D:** For each target zone (proximal neck, distal left, distal right) the interface shows the 3D model as seen from the suggested LAO/RAO and CRA/CAU angles, with the landing plane drawn on it. Sliders let the surgeon refine the projection before saving it to the plan.
- **The plan stays consistent:** When the surgeon approves the measurements, a modal shows the suggested main body, contralateral limbs and iliac extenders, with the overlap, covered length and remaining distance to the target for each. The configuration is always computed from the current values, so any edited measurement is taken into account automatically.

Here are some screenshots from the interface:

<figure>
  <img src="{{ '/projects/images/surgical-planning-app-measurements.jpg' | relative_url }}" alt="Measurement editor showing an axial cross-section of the straightened vessel with the maximum and minimum diameters drawn on it, next to coronal and sagittal views">
  <figcaption>Measurement review: the axial cross-section of the straightened vessel with the maximum (red) and minimum (yellow) diameters, which the surgeon can move, plus coronal and sagittal views.</figcaption>
</figure>

<figure>
  <img src="{{ '/projects/images/surgical-planning-app-carm.jpg' | relative_url }}" alt="3D model of the aorta and renal arteries seen from the suggested C-arm angle, with sliders for left/right and head/foot angulation">
  <figcaption>C-arm planning: the 3D model seen from the suggested projection for the proximal neck, with the landing plane and sliders to adjust the angles.</figcaption>
</figure>

<figure>
  <img src="{{ '/projects/images/surgical-planning-app-stentgraft.jpg' | relative_url }}" alt="Modal window listing the suggested stent-graft main body, compatible contralateral limbs and extenders, with lengths and distances to target">
  <figcaption>Stent-graft configurator: suggested main body, compatible contralateral limbs and extenders, recomputed whenever a measurement changes.</figcaption>
</figure>

### How the inteface is built

The interface is a **React** single-page app talking to a **FastAPI** backend service. The heavy processing is done in the pipeline explained before; the web service only reads its outputs (volumes, masks, meshes, measurement tables) and answers interactive requests.

## What I've learned from this

Through this project I learned how to manage the entire process of developing a deep learning-based tool, from conducting initial research to building a functional MVP. I gained experience working with medical imaging data (which I never worked with), developing and validating deep learning models for a real case study, and understanding how to translate model outputs into meaningful clinical information. I also learned the importance of working in a multidiscplinary team: meeting and designing software with clinicians', making sure that the results are not only technically reliable but also useful and interpretable for them in a real clinical setting!