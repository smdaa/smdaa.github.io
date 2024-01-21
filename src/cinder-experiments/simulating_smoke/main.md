# Simulating Smoke

The following article is an implementation and a summarization of this paper:

[Real-time smoke simulation](https://repository.bilkent.edu.tr/items/8a55d570-1188-4bab-b938-72683f0b9464)

Link to source code on github : [Simulating Smoke Source Code](https://github.com/smdaa/creative-coding/blob/main/src/example_1/main.cpp)

## Theoretical background
### Navier-Stokes equations
The Navier–Stokes equations are a set of partial differential equations that describe the motion of viscous fluid substances. In three-dimensional form, they are written as follows:

* The momentum equation:
\\[
\frac{\partial \vec{u}}{\partial t} + \vec{u} \cdot \nabla \vec{u} + \frac{1}{\rho} \nabla p = \vec{f} + \nu \nabla^2 \vec{u}
\\]

* Incompressibility Condition:
\\[
\nabla \cdot \vec{u} = 0
\\]

where:
- \\(\vec{u}\\) represents the velocity vector field of the fluid.
- \\(t\\) is the time variable.
- \\(\rho\\) is the fluid density.
- \\(p\\) denotes the pressure.
- \\(\vec{f}\\) is the force vector applied on the fluid.
- \\(\nu\\) is the kinematic viscosity of the fluid.

Well, what does all that means for us non physicists?
* The momentum equation:

The term \\(\frac{\partial \vec{u}}{\partial t}\\) represents how the velocity of the fluid changes with time.

The \\(\vec{u} \cdot \nabla \vec{u}\\) term reflects the idea that the fluid is carrying itself along. Picture yourself in a boat on a river; this part of the equation describes how the current carries you downstream.

The \\(\frac{1}{\rho} \nabla p \\) term is like the push from differences in pressure. Think about how wind pushes air around or how water moves in response to changes in pressure.

The \\(\vec{f}\\) term accounts for the influence of an external force, for example gravity. In a waterfall, for example, gravity is pulling the water downward.

The \\(\nu \nabla^2 \vec{u}\\) term deals with viscosity, which is how "sticky" or resistant to flow the fluid is. Consider stirring honey; the viscosity of honey would resist the motion of your spoon. This term quantifies how the fluid resists changes in its flow.

* Incompressibility Condition:

The incompressibility condition \\(\nabla \cdot \vec{u} = 0\\) is like saying the fluid is incompressible, meaning it doesn't get squeezed or expanded. Picture a balloon filled with water. When you squeeze one part of the balloon, water immediately moves to another part to maintain the constant overall volume.

### Smoke model
The Helmholtz-Hodge Decomposition states that any vector field \\(\vec{w}\\) can be decomposed into the form:
\\[
\vec{w} = \vec{u} + \nabla q    
\\]
where \\(\nabla \vec{u} = 0\\) (\\(\vec{u}\\) is divergence free) and \\(q\\) is a scalar field. This outcome allows us to define an operator \\(\mathbf{D}\\) which projects any vector field w onto its divergence free part: \\(\vec{u} = \mathbf{D} \vec{w}\\)

Let's project the both sides of the momentum equation (keeping in mind the incompressibility condition ie \\(\nabla \cdot \vec{u} = 0\\)):
\\[
\frac{\partial \vec{u}}{\partial t} = - \vec{u} \cdot \nabla \vec{u} + \vec{f} + \nu \nabla^2 \vec{u}
\\]
The pressure term disappeared because it is a gradient field.

The equation above is the basis of our smoke simulation.

### Thermal Buoyancy
Thermal buoyancy refers to the upward force exerted on an object immersed in a fluid (liquid or gas) due to temperature differences within that fluid. 
\\[
F_{buoy} = (\alpha d - \beta(T - T_{amb})) \vec{g}
\\]

where:
- \\(d\\) represents smoke concentration.
- \\(\alpha\\) and \\(\beta\\) are two constants.
- \\(T\\) is the temperature at the current cell.
- \\(T_{amb}\\) is the average temperature of the fluid grid.
- \\(\vec{g}\\) is is the gravity.

In other words smoke is hotter than air that's why it gets pushed up.

### Turbulence
Turbulence refers to a swirling-like motion which represents how much the velocity field rotates around a point. So naturally it is represented with the curl operator:
\\[
\vec{w} = \nabla \times \vec{u}
\\]

Nevertheless, in the calculation of the turbulence equation (not shown here), the flow disperses, leading to the loss of some turbulence. To restore a portion of the lost turbulence, a new technique, turbulence confinement, is introduced:

\\[
F_{vortCof} = \frac{\nabla || \vec{w} ||}{|| \nabla || \vec{w} || ||} \times \vec{w}
\\]

The core concept is to identify the positions of vortices and apply a body force to enhance the rotational motion around each vortex or cell.