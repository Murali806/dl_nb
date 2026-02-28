Question => we have 2 mics seperated by distance d horizontally.  a speech source is perpendicular/vertically above mic 1 by same distance d, this means theta is 90degree?  and mic 2 theta will be  45degrees? as distance is same.

## Great Question! Let me visualize this:

```
Speech Source (S)
      |
      | d (vertical)
      |
    Mic1 -------- Mic2
         d (horizontal)
```

### For this geometry:

**You are correct that θ is different for each mic!**

This is because θ is measured as the **angle of the incoming wave at each mic's perspective**... 

Actually wait - let me be precise:

In **beamforming**, θ is typically defined as a **single global angle** from the array axis, not per-mic. So:

```
Speech Source (S)
      |  \
      |    \
      |  θ   \
    Mic1 ---- Mic2
```

- The angle **from Mic1** to source = 90°
- The angle **from Mic2** to source = 45° (since equal sides d, d)
- The **global array angle θ** = somewhere in between

### The key distinction:

| Approach | θ per mic? |
|----------|-----------|
| **Far-field assumption** (standard beamforming) | ❌ Same θ for all mics (plane wave) |
| **Near-field beamforming** | ✅ Different angle per mic |

### Your example is actually a **near-field** scenario!
Because the source is **close** to the array (distance ≈ mic spacing), so the **wavefront is curved**, not a flat plane.

In **far-field**, the source is assumed to be **very far away**, so all mics see the **same θ** with parallel wavefronts. 🎯

-------------------------------------------------------------------------------------

in my example above - lets say  distance  b/w mic1 ans speech source is huge say 100*distance and   distance b/w mic 1 and mic2 is say distance.   In this case still theta should be 90 and 45 correct?

## No! This is exactly the Far-field magic! 🎯

```
Speech Source (S)
      |
      | 100d (very far)
      |
    Mic1 -------- Mic2
         d (horizontal)
```

### When source is very far away:

The **tiny horizontal shift** of `d` between mics becomes **negligible** compared to `100d` vertical distance.

```
Speech Source (S)
      |    \
      |      \  ← this angle difference
      |        \    becomes negligible!
      | 100d  100d (≈same)
      |          \
    Mic1 ---d--- Mic2
```

### The math:
- Angle at Mic1 = arctan(100d/0) = **90°**
- Angle at Mic2 = arctan(100d/d) = **89.4°** ← almost same!

As distance → ∞:
$$\theta_{mic1} \approx \theta_{mic2} \approx \theta_{global}$$

### Intuition:
Think of **sunlight** ☀️ - sun is so far that all rays hitting earth are **parallel**, same angle everywhere. If sun were close, angles would differ per location.

| Source Distance | Wavefront | θ per mic |
|----------------|-----------|-----------|
| Close (near-field) | Curved 🌊 | Different |
| Far (far-field) | Flat/Plane 📏 | **Same** ✅ |

So in your **100d example**, both mics see **≈ same θ** → Far-field assumption holds! 🎯

--------------------------------------------------------
Question => ## For your 100d example:

```
Speech Source (S)
      |
      | 100d (vertical)
      |
    Mic1 -------- Mic2
         d (horizontal)
```

### θ is measured from the **array axis** (horizontal):

```
    Mic1 -------- Mic2
      \  θ
        \
          \
           S (if source were at an angle)
```

But in your case source is **directly above Mic1**, so:

$$\theta = 90°$$

from the array axis (horizontal line between mics)

### Plugging into delay formula:

$$\tau = \frac{d \cdot \sin(\theta)}{c} = \frac{d \cdot \sin(90°)}{c} = \frac{d \cdot 1}{c} = \frac{d}{c}$$

### What this means:
- Sound hits **Mic1 first** (it's directly below source)
- Mic2 receives the sound **d/c seconds later**
- This makes intuitive sense! The horizontal separation `d` is the extra path Mic2 sound has to travel

The `100d` vertical distance **doesn't matter** for the delay calculation in far-field - only the **horizontal component** `d·sin(θ)` matters! 🎯for my example => asper your time delay formulae => what is theta value?

