+++
title = "Building a dynamic particle system"
date = 2024-01-13
tags = ["libcinder", "cpp", "creative coding"]
aliases = ["/cinder-experiments/building_a_dynamic_particle_system/main.html"]
+++

![](/assets/building-a-dynamic-particle-system/final.webp)

[View source code on GitHub](https://github.com/smdaa/creative-coding/blob/main/src/example_1/main.cpp)

## Table of contents

## Creating a particle system

Let's start by creating the particle object. A particle has the following properties: a radius $r$, a position $(x, y)$, a velocity $(v_x, v_y)$, and an optional color.

```cpp
class Particle {
 public:
  vec2 position;
  vec2 velocity;
  float radius;
  Color color;

  Particle(const vec2 &position, const vec2 &velocity, float radius,
           Color color)
      : position(position), velocity(velocity), radius(radius), color(color) {}

  void draw() {
    gl::color(color);
    gl::drawSolidCircle(position, radius);
  }
  void updateVelocity(vec2 force, float dt) { velocity += force * dt; }
  void updatePosition(float dt) { position += velocity * dt; }
};
```

Since we want our particle to remain inside the screen, we will use the factor **WALL_BOUNCE_FACTOR** $\in [0, 1]$ to bounce the particle in the other way if it leaves the screen.

```cpp
class Particle
{
public:
  vec2 position;
  vec2 velocity;
  float radius;
  Color color;

  ...

  void checkEdgeCollision() {
    if (position.x - radius < 0) {
      position.x = radius;
      velocity.x = -WALL_BOUNCE_FACTOR * velocity.x;
    } else if (position.x + radius > WINDOW_WIDTH) {
      position.x = WINDOW_WIDTH - radius;
      velocity.x = -WALL_BOUNCE_FACTOR * velocity.x;
    }

    if (position.y - radius < 0) {
      position.y = radius;
      velocity.y = -WALL_BOUNCE_FACTOR * velocity.y;
    } else if (position.y + radius > WINDOW_HEIGHT) {
      position.y = WINDOW_HEIGHT - radius;
      velocity.y = -WALL_BOUNCE_FACTOR * velocity.y;
    }
  }
};
```

Let's simulate the collision of two particles next. The model is quite basic; it assumes that the two particles have the same mass and that the collision is elastic.

```cpp
class Particle
{
public:
  vec2 position;
  vec2 velocity;
  float radius;
  Color color;

  ...

  void checkParticleCollision(Particle &other) {
    vec2 relativePosition = other.position - position;
    vec2 relativeVelocity = other.velocity - velocity;
    float distance = glm::length(relativePosition);
    float combinedRadius = radius + other.radius;

    if (distance < combinedRadius) {
      // Particles are colliding
      float penetration = combinedRadius - distance;
      vec2 collisionNormal = glm::normalize(relativePosition);
      float relativeVelocityAlongNormal =
          glm::dot(relativeVelocity, collisionNormal);
      if (relativeVelocityAlongNormal > 0) {
        return;
      }
      float j = -relativeVelocityAlongNormal;
      vec2 impulse = j * collisionNormal;
      velocity -= impulse;
      other.velocity += impulse;
      position -= 0.5f * penetration * collisionNormal;
      other.position += 0.5f * penetration * collisionNormal;
    }
  }
};
```

## Creating a 2D world

We will have a list of particles in our environment, and we will need to establish an energy field in order to move them.

Using the **GRID_RESOLUTION** parameter, we will divide our 2D plane into a grid. For each coordinate in the grid $(x_i, y_i)$, we will have an angle pointing in the direction of motion. We'll make the field point in the direction of the mouse to make the experience interactive.

```cpp
class World {
 public:
  float dt;
  int gridNumRows;
  int gridNumCols;
  std::vector<std::vector<float>> grid;
  std::vector<Particle> particles;

  World() {
    dt = DT;
    gridNumRows = WINDOW_HEIGHT / GRID_RESOLUTION;
    gridNumCols = WINDOW_WIDTH / GRID_RESOLUTION;
    grid = std::vector<std::vector<float>>(
        gridNumRows, std::vector<float>(gridNumCols, 0.0f));
    particles = std::vector<Particle>();
  }

  void addParticle(const Particle &particle) { particles.push_back(particle); }

  void updateGrid(const vec2 &mousePos) {
    for (int i = 0; i < gridNumRows; ++i) {
      for (int j = 0; j < gridNumCols; ++j) {
        float x = static_cast<float>(j) * static_cast<float>(GRID_RESOLUTION);
        float y = static_cast<float>(i) * static_cast<float>(GRID_RESOLUTION);

        grid[i][j] = atan2(mousePos.y - y, mousePos.x - x);
      }
    }
  }

};
```

{{< video src="/assets/building-a-dynamic-particle-system/grid.webm" type="video/webm" width="800" >}}

The process of updating the particles is simple: using the factor **FORCE_FEILD_FACTOR**, each particle is moved towards the angle of the closest point in the 2D grid. To add some variation, the force of movement is proportional to the radius. After that, we look for collisions.

```cpp
class World {
 public:
  float dt;
  int gridNumRows;
  int gridNumCols;
  std::vector<std::vector<float>> grid;
  std::vector<Particle> particles;

  ...

  void updateParticles(float dt) {
    for (int i = 0; i < particles.size(); ++i) {
      int rowIndex = static_cast<int>(std::min(
          std::max(
              particles[i].position.y / static_cast<float>(GRID_RESOLUTION),
              0.0f),
          static_cast<float>(gridNumRows - 1)));
      int columnIndex = static_cast<int>(std::min(
          std::max(
              particles[i].position.x / static_cast<float>(GRID_RESOLUTION),
              0.0f),
          static_cast<float>(gridNumCols - 1)));
      float gridValue = grid[rowIndex][columnIndex];
      vec2 force = FORCE_FEILD_FACTOR * particles[i].radius *
                   vec2(cos(gridValue), sin(gridValue));
      particles[i].updateVelocity(force, dt);
      particles[i].updatePosition(dt);
      for (auto &other : particles) {
        if (&particles[i] != &other) {
          other.checkParticleCollision(particles[i]);
        }
      }
      particles[i].checkEdgeCollision();
    }
  }
};
```

{{< video src="/assets/building-a-dynamic-particle-system/particles.webm" type="video/webm" width="800" >}}

## Drawing

Instead of just drawing the particles let's create a triangular mesh.
The mesh is created by extruding the particle positions along the z-axis and connecting them to form triangles.
The use of alpha blending ensures smooth transitions especially when the number of particles is big.

```cpp
class CollisionApp : public App {
 public:

  ...

  void draw() override {
    if (world.particles.size() > 2) {
      // Create a TriMesh
      TriMesh::Format format = TriMesh::Format().positions(3);
      TriMesh mesh(format);

      // Extrude the path
      float extrusionDepth = 10.0f;
      for (const auto &particle : world.particles) {
        vec3 position(particle.position, 0);
        mesh.appendPosition(position);

        position.z += extrusionDepth;
        mesh.appendPosition(position);
      }

      // Add triangles
      for (size_t i = 0; i < mesh.getNumVertices() - 4; i += 4) {
        mesh.appendTriangle(i, i + 2, i + 4);
        mesh.appendTriangle(i + 1, i + 3, i + 5);
      }

      // Draw the mesh
      gl::color(MESH_COLOR);
      gl::enableAlphaBlending();
      gl::draw(mesh);
      gl::disableAlphaBlending();
    }
  }

```

### Example 1

The following is an example of the result with a low count of particles:

{{< video src="/assets/building-a-dynamic-particle-system/example1.webm" type="video/webm" width="800" >}}

### Example 2

The following is an example of the result with a high count of particles:

{{< video src="/assets/building-a-dynamic-particle-system/example2.webm" type="video/webm" width="800" >}}

you can experiment with the different variables and see what you will get:

```cpp
#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080
#define NPARTICLES 500
#define MAX_RADIUS 5.0f
#define MIN_RADIUS 1.0f
#define MAX_VELOCITY 0.0f
#define MIN_VELOCITY 0.0f
#define WALL_BOUNCE_FACTOR 0.2f
#define FORCE_FEILD_FACTOR 0.5f
#define DT 0.9f
#define GRID_RESOLUTION 5
#define DRAW_GRID false
#define DRAW_PARTICLES false
#define BG_COLOR Color(0.6f, 0.6f, 0.6f)
#define MESH_COLOR ColorA(0.0f, 0.0f, 0.0f, 0.05f)
```
