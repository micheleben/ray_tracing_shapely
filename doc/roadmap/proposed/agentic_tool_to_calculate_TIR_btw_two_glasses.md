# Agentic Tool to calculate TIR and TIR physics between two glasses
## Code Style
When implementing this roadmap please follow the code style as described in CONTRIBUTING.md

## Aim
We want a tool an agent can call to calculate the TIR angle between two glasses of the scene. 

The tool should also be a general tool we can use to see how the relative phase shift between polarizations varies approaching the TIR angle and over the TIR angle.

### the way I envisioned the tool:

It should have as input parameters: 
* the two names of the glasses, we ask the user to put the first glass as the one with higer index of refraction.
* optionally the angle of incidence between the glass with higher index and the glass with lower index.
* optionally a delta_angle to describe angles that are close to the TIR or close to BREWSTER angles. default 1 degree.

The code should do the following behaviour:

* It should check which one has the greater index of refraction and return an agentic friendly error if the first prism is not the one with the highest index

* It should return the following physical quantities:
    + some flags to understand which output the agent is seing (covering the different options we see below and errors as we described above)
    + the TIR angle
    + if the angle is provided:
        - if it produces refraction calculate all the refraction quantities ('T_s', 'T_p', 'ratio_Tp_Ts','R_s','R_p'). Calculate the phase difference between s and p even if it is trivial. Flag this with "angle < TIR" or similar
        - if it does not produce refraction return "not applicable" or similar for the refraction quantities and calculate the phase shifts. Flag this with "angle > TIR"
        - special flag for angle within +/- delta_angle (default +/- 1 degree) to the TIR angle
        - special flag for angle within +/- delta_angle (default +/- 1 degree) to the BREWSTER angle
    + if the angle is not provided
        - flag this as "default calcs at + delta_angle degree from TIR" and return calculations for an angle that is + delta_angle from the TIR      

## Physics Background
### 1. Phase Shifts Before TIR

(\(\theta _{i}<\theta _{c}\))

In this regime, the reflection coefficients are real numbers. A real value means the phase shift 
(\(\delta \))
can only be: 
* \(\delta =0\): The reflected wave is in phase with the incident wave (positive coefficient).
 
* \(\delta =\pi \) (\(180^{\circ }\)): The reflected wave is perfectly out of phase (negative coefficient). 
 
For internal reflection (moving from a denser to a rarer medium), the phase behavior is as follows: 
 
 * s-polarization: Typically has no phase shift 
 
 (\(\delta _{s}=0\)) 
 
 for all angles below the critical angle.
 
 p-polarization: Has a \(\pi \) phase shift at low angles, but this switches to \(0\) once you surpass the Brewster angle. 
 
 ### 2. Phase Shifts During TIR 
 (\(\theta _{i}>\theta _{c}\)) 
 
 Once you surpass the critical angle, the reflection coefficients become complex numbers. 
 This results in continuous, non-trivial phase shifts 
 
 (\(\delta _{s},\delta _{p}\)) 
 
 that depend on the specific angle of incidence. The Formulas for Phase Shifts in TIR The phase advances for the two components are given by these Fresnel-derived formulas:
 
  \(\tan \left(\frac{\delta _{s}}{2}\right)=\frac{\sqrt{n^{2}\sin ^{2}\theta _{i}-1}}{n\cos \theta _{i}}\)\(\tan \left(\frac{\delta _{p}}{2}\right)=\frac{n\sqrt{n^{2}\sin ^{2}\theta _{i}-1}}{\cos \theta _{i}}=n^{2}\tan \left(\frac{\delta _{s}}{2}\right)\)
 
 \(n=n_{incident}/n_{transmitted}\): The relative refractive index (e.g., \(1.5\) for glass-to-air).
 \(\theta _{i}\): The angle of incidence.
 \(\delta _{s},\delta _{p}\): The absolute phase shifts for s- and p-polarized light. 
 
 #### 3. The Relative Phase Shift 
 (\(\Delta \)) 
 The most important value for polarization is the differential phase shift 
 (\(\Delta =\delta _{p}-\delta _{s}\)). 
 
 This difference is what turns linear polarization into elliptical polarization: 
 
 \(\tan \left(\frac{\Delta }{2}\right)=\frac{\cos \theta _{i}\sqrt{n^{2}\sin ^{2}\theta _{i}-1}}{n\sin ^{2}\theta _{i}}\)
 
 ### Summary of Differences Phase Characteristic
  
 | Region | Angles | Phase shift Nature | Coefficients | Results | Slope |
 |:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
 | Below Critical Angle | (\(\theta _{i}<\theta _{c}\)) | Trivial: Discrete values (\(0\) or \(\pi \)) | Real Coefficients | Reflected light stays linear | Constant between transitions |
 | Above Critical Angle |  (\(\theta _{i}>\theta _{c}\)) | Non-Trivial: Continuous variation | Complex numbers | Reflected light becomes elliptical | Infinite slope at the critical angle |

## Status of the codebase

we have in analysis.fresnel_utils

```python
def fresnel_transmittances(
    n1:float, n2:float, theta_i_deg:float
)->Dict[str, float]:
```
```python
Compute Fresnel power transmittances and reflectances at an interface.

Uses the standard Fresnel equations for a planar interface between
two dielectric media.

Args:
    n1: Refractive index of the incident medium.
    n2: Refractive index of the transmitting medium.
    theta_i_deg: Angle of incidence in degrees (from normal).

Returns:
    Dict with keys:
    - 'T_s': s-polarization power transmittance
    - 'T_p': p-polarization power transmittance
    - 'R_s': s-polarization power reflectance
    - 'R_p': p-polarization power reflectance
    - 'ratio_Tp_Ts': T_p / T_s (or float('inf') if T_s ~ 0)
    - 'theta_t_deg': refraction angle in degrees

Raises:
    ValueError: If the angle exceeds the critical angle (TIR).

```
It is a great one to study refraction , here we want to study reflectance.



## Proposed changes
Few touch_points:
1. add a tool in analysis.fresnel_utils that implements the physic we discussed above. I understand part of the physics overlap with analysis.fresnel_utils but we will likely keep both.
2. JSON schema additions
3. add the tool in agentic_tools.py
4. update list of tools in tool_registry.py
