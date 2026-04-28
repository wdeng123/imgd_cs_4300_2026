# Towards Boids on the GPU

This tutorial aims to get us at least part of the way towards implementing 
the [Boids (aka Flocking) algorithm](http://www.kfish.org/boids/pseudocode.html), 
originally presented [in a classic paper that has been cited over 11000 times](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=2ahUKEwiqy5OP_tPoAhWDuJ4KHdqmDH4QFjACegQIIhAB&url=http%3A%2F%2Fwww.cs.toronto.edu%2F~dt%2Fsiggraph97-course%2Fcwr87%2F&usg=AOvVaw2StOMrXs0E_nHLgD87UrGN). 
[Here’s a video of the original simulation in action](https://www.youtube.com/watch?v=86iQiV3-3IA). 
For an in-depth treatment of agent-based steering algorithms more generally (including Boids) 
try [this chapter from the Nature of Code](https://natureofcode.com/book/chapter-6-autonomous-agents/).  T
he author of this chapter also walks through implementing the algorithm in 
[his amazing Coding Train series](https://www.youtube.com/watch?v=mhjuuHl6qHM). 
But the first link in this assignment is perhaps the easiest to follow, with clear pseudocode
and explanations of the algorithm. 
[Here is a good article on the history of the algorithm](https://beforesandafters.com/2022/04/07/a-history-of-cg-bird-flocking/).



In some ways, this simulation is simpler than our previous Physarum simulation, as we don't have an
underlying continuous layer to deal with (unless you want to add one that influences the agents in some way).
But our complexity now becomes that boids need to know about each other in a way we haven't explored before.
In the game of life, cells needed to know about their immediate neighbors; 
here every agent needs to (potentially) know about every other agent in the flock.

This means we need a way to loop through the flock. A `for` loop is all we need for this.

Remember that if, for every given agent A, we're looking up all other agents stored in group A\*
of size 8192, that means processing all agents will require 8192*8192 buffer lookups... 67 million!
And that's not even including all the actual vector math that is performed once 
we have the position of each agent. That said, on even a moderately new integrated graphics card
you can expect a 15-20x speedup by running boids on the GPU instead of the CPU. 
For fancy graphics cards (1080, 2080, 3060 on up) you can expect even more than this; definitely worth
the effort.

OK, here we go!

## The Setup

The [template project we'll use for this tutorial](https://github.com/charlieroberts/seagulls/tree/main/howtos/13_many_particles) is the seagulls `many_particles` howto. 
We don't need to do anything to our `main.js` file or our `render.glsl` file for now; everything is all set to go there.
We just need to modify our compute shader. Once boids is up and running, we'll play around with
the rendering / number of agents.

Let's go ahead and open up our `compute.glsl` file. Make sure you change the `Particle` definition at the top of
file to use a `vec2f` for velocity instead of a single float for `speed`.

```wgsl
struct Particle {
  pos: vec2f,
  vel: vec2f
};

@group(0) @binding(0) var<uniform> res:   vec2f;
@group(0) @binding(1) var<storage, read_write> state: array<Particle>;

fn cellindex( cell:vec3u ) -> u32 {
  let size = 8u; // 8 is our workgroup size on one axis
  return cell.x + (cell.y * size);
}

@compute
@workgroup_size(8,8,1)

fn cs(@builtin(global_invocation_id) cell:vec3u)  {
  let count:u32 = 2048u;

  let idx            = cellindex( cell );
  var boid:Particle  = state[ idx ];

  var center:vec2f   = vec2f(0.); // rule 1

  for( var i:u32 = 0u; i < count; i = i + 1u ) {
    // don't use boids' own properties in calculations
    if( idx == i ) { continue; }

    let _boid = state[ i ];
    
    // rule 1
    center += _boid.pos;
  }

  // apply effects of rule 1
  center /= f32( count - 1u );
  boid.vel += (center-boid.pos) * .05;
  
  // calculate next position
  boid.pos = boid.pos + (2. / res) * boid.vel;

  // boundaries, flip velocity if you pass one
  if( abs( boid.pos.y ) >= 1. ) { 
    boid.vel.y *= -1.; 
  }
  if( abs( boid.pos.x ) > 1. ) {
    boid.vel.x *= -1.;
  }

  // make sure to reassign back to buffer
  // otherwise no effect will occur!
  state[ idx ] = boid;
}
```

In the above code we implement rule 1 for boids, which tells each boid to move towards 
a center position by averaging the position of all other boids and using this value to
influence the boids velocity. We also invert the velocity whenever we cross a border (-1 or 1)
on the x and y axes.

It's the same basic process for rules two and three, just with a bit different math. Rule 2 will 
tell each boid to try and keep a minimum distance from other boids, while Rule 3 will tell boids
to try and match velocities.

```wgsl
struct Particle {
  pos: vec2f,
  vel: vec2f
};

@group(0) @binding(0) var<uniform> frame: f32;
@group(0) @binding(1) var<uniform> res:   vec2f;
@group(0) @binding(2) var<storage, read_write> state: array<Particle>;

fn cellindex( cell:vec3u ) -> u32 {
  let size = 8u;
  return cell.x + (cell.y * size) + (cell.z * size * size);
}

@compute
@workgroup_size(8,8)

fn cs(@builtin(global_invocation_id) cell:vec3u)  {
  let count:u32 = 2048u;

  let idx            = cellindex( cell );
  var boid:Particle  = state[ idx ];

  var center:vec2f   = vec2f(0.); // rule 1
  var keepaway:vec2f = vec2f(0.); // rule 2
  var vel:vec2f      = vec2f(0.); // rule 3

  for( var i:u32 = 0u; i < count; i = i + 1u ) {
    // don't use boids' own properties in calculations
    if( idx == i ) { continue; }

    let _boid = state[ i ];
    
    // rule 1
    center += _boid.pos;
    
    // rule 2
    if( length( _boid.pos - boid.pos ) < .005 ) {
      keepaway = keepaway - ( _boid.pos - boid.pos );
    }
    
    // rule 3
    vel += _boid.vel;
  }

  // apply effects of rule 1
  center /= f32( count - 1u );
  boid.vel += (center-boid.pos) * .05;

  // apply effects of rule 2
  boid.vel += keepaway;

  // apply effects of rule 3
  vel /= f32( count - 1u );
  boid.vel += vel * .01;
  
  // calculate next position
  boid.pos = boid.pos + (2. / res) * boid.vel;

  // boundaries
  if( abs( boid.pos.y ) >= 1. ) { 
    boid.vel.y *= -1.; 
  }
  if( abs( boid.pos.x ) > 1. ) {
    boid.vel.x *= -1.;
  }

  state[ idx ] = boid;
}
```

Last but not least, let's implement one of the extra rules that Reynolds describes. In this case
we'll limit the speed of our boids so that they don't go haywire. You can put this fragment right before the 
next position is calculated in the compute shader.

```wgsl
// limit speed
if( length( boid.vel ) > 5. ) {
  boid.vel = (boid.vel / length(boid.vel)) * 5.;
}
```

That's it! You should have a reasonable setup for boids at this point, now you can play around with the rendering,
the number of agents, and the various coefficients used in the algorithm. You might also try exploring some
of the other optional rules of boids described in the Reynolds article.

## Make it prettier

Next let's make the render a bit nicer. First, let's pass in a set of triangle vertices to represent each particle in our `main.js` file. Add the following line to your render pass definition:

`vertices:seagulls.constants.shapes.triangle,`

This will change from the default quad to a triangle for each particle. Our vertex shader won't need to change much, but we'll add a function that performs 2D rotation and then set our triangles to point in the direction indicated by their current velocity. Replace the vertex shader in the `render.wgsl` file with the following... but make sure to not overwrite the fragment shader!

```wgsl
struct VertexInput {
  @location(0) pos: vec2f,
  @builtin(instance_index) instance: u32,
};

struct Particle {
  pos: vec2f,
  vel: vec2f
};

@group(0) @binding(0) var<uniform> frame: f32;
@group(0) @binding(1) var<uniform> res:   vec2f;
@group(0) @binding(2) var<storage> state: array<Particle>;

fn rotate(p:vec2f, a:f32) -> vec2f {
  let s = sin(a);
  let c = cos(a);
  let m = mat2x2(c, s, -s, c);
  return m * p;
}

@vertex 
fn vs( input: VertexInput ) ->  @builtin(position) vec4f {
  let v = state[ input.instance ];
  let p1 = input.pos * .02;
  // only three vertices so who cares I guess?
  let a = atan2(v.vel.x, v.vel.y);
  let p = rotate(p1, a);
  let aspect = (res.y / res.x);
  return vec4f( v.pos.x - p.x * aspect, v.pos.y + p.y, 0., 1.); 
}
```

That should be enough to get triangles moving around the screen, and pointing in the direction that they're heading.

## Optimizing via spatial hashing
This implementation of boids is O(n2) complexity, so, we'll need to do some optimizations if we want to improve the number of agents we can run. On my five-year-old computer with integrated graphics, the implementation we have so far can do a couple thousand agents... which is still *way* better than what it would do on the CPU. But, with spatial hashing, we'll be able to get that up to a couple hundred thousand at least.

The idea of spatial hashing in boids is simple: instead of using the position and velocity of *every other boid* in the simulation to determine each boid's movement, only compare a given boid to boids that are *close to it*. To optimize this further we want to make the "close" boids occupy contiguous memory, so that their data will be more likely to be cached. 

Spatial hashing (aka binning) takes care of this. An overview is:

1. Determine a grid *resolution*, or how large the space of neighbors should be. For a fullscreen boids simulation a 60x60 grid should yield enough neighbors per bin.
2. Calculate the number of boids in each bin, in order to determine index offsets. That is, if we know a boid should go into bin #2, and we also know that there four boids in bin 0 and 3 in bin 1, then the boids of bin #2 should start at index 7.
3. Reorganize the memory holding the boids by bin.

Steps 2 & 3 are performed *per frame*, or per tick of the simulation, in separate compute shaders. An additional step, *prefix summing*, enables us to optimize the algorithm further, but even completing steps 2 & 3 alone will yield significant performance improvements.

The code in this section is loosely based off [this fantastic implementation / explanation of Particle Life](https://lisyarus.github.io/blog/posts/particle-life-simulation-in-browser-using-webgpu.html)

### Calculate the number of boids per bin
Our first shader will determine the bin of each boid, and increment the appropriate `binSize` value at its index. We'll use *atomics* to do this, which both read and write to a given memory location (only u32 values are supported) in the same hardware operation; this prevents race conditions from occurring where multiple shader invocations are trying to update the same value at the same time.

Create a new shader and name it `binsize_compute.wgsl` with the following code.

```wgsl
struct Particle {
  pos: vec2f,
  vel: vec2f
};

const GRID_SIZE:i32 = 60;

@group(0) @binding(0) var<storage> particles : array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles2: array<Particle>;
@group(0) @binding(2) var<storage, read_write> binSize : array<atomic<u32>>;

@compute @workgroup_size(64,1,1)
fn cs(@builtin(global_invocation_id) id : vec3u) {
  if (id.x >= arrayLength(&particles)) { return; }

  // Read the particle data
  let particle = particles[id.x];

  // Compute the linearized bin index
  let binIndex = getBinIndex( particle.pos );

  // Increment the size of the bin
  atomicAdd(&binSize[binIndex], 1u);
}

fn getBinIndex(position : vec2f) -> i32 {
  // position is -1,1, offset by one and scale by half the bin size
  let binxy = vec2i( 
    i32( (1+position.x) * f32(GRID_SIZE/2) ),
    i32( (1+position.y) * f32(GRID_SIZE/2) ) 
  );

  return binxy.y * GRID_SIZE + binxy.x;
}
```

In your `main.js` file,  create a new buffer to store the bin sizes: 

```js
const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

const sizes_b  = sg.buffer( new Float32Array(GRID_SIZE*GRID_SIZE), '', usage )
```

...where `GRID_SIZE` represents one dimension of your grid (e.g. 60). We're also adding some extra usage flags to this buffer so we can read from it on the CPU, this will enable us to look at the contents and see if it's working. Next we'll setup our shader pass, after also creating a second state buffer to pingpong:

```js
const binsize_shader = await seagulls.import( './binsize_compute.wgsl' )
const state_b2 = sg.buffer( state )

const sizes = sg.compute({
  shader: binsize_shader,
  data:[
    sg.pingpong(state_b,state_b2),
    sizes_b
  ],
  onframe() { sizes_b.clear() },
  dispatchCount:[NUM_PARTICLES / (WORKGROUP_SIZE * WORKGROUP_SIZE),1,1]
})
```

Note that we'll clear (setting all values to zero) the sizes buffer on every frame, as we're determining the bin positions of our boids every time the simulation runs. Before we start running the shader, let's test it by looking at the values it generates after a single frame. To do  this, first comment out the `sg.run( computer, render )` line at the bottom of `main.js`, and replace it with the following:

```js
await sg.once( sizes, compute, render )
console.log( await sizes_b.read( null, 0, Uint32Array ))
```

The `buffer.read()` function accepts three values, a size value for how much of the buffer should be read, an offset, and the type of array to put the buffer data into. Passing a value of `null` for size and `0` for offset means the whole buffer will be read.

If you run this now with 2048 particles and a grid size of 60, you should see one or two (or zero) particles per bin when you look in the log. This makes sense given a random distribution and 2048 particles spread out across 60x60 (3600) bins. Try changing `GRID_SIZE` to be `4` in the shader and rerunning. Now you'll see that all the particles have been summed across the first 16 indices of our buffer. It looks like the shader is working! 

A couple of notes:
1. There aren't that many boids per bin with a 60x60 grid size. But we'll actually be looking at multiple bins per boid, including both the bin the boid resides in and all the neighbor bins.
2. We're going to run hundreds of thousands of bods, so potentially there will be up to 50x as many boids per bin when we increase the particle count.

### Placing the boids
Now that we know the number of boids for each bin, we can organize a buffer by bin. That is, we'll store all the boids for bin 0 first, and then all the boids for bin 1, then bin 2 etc. The only reason we're able to do this is because we know all the sizes for each bin, which gives us an offset for where we should place each particle. We'll also need to keep track of how many particles we've added to the current bin (via atomics) in order to determine the final index for each boid.

#### Prefix summing
To make this faster, we need to calculate the offsets for each bin in advance. For example, given bin sizes of:

0: 20
1: 10
2: 30

... the starting index for bin 3 would be 60. Calculating these starting indices in a shader pass and storing the results will make it much quicker to look them up throughout the rest of our simulation. This algorithm is known as [prefix summing](https://www.reddit.com/r/leetcode/comments/1ewxiqt/a_visual_guide_to_prefix_sums/) , while parallel versions of this exist we'll just do it serially for now as it's not going to be a significant cost in our simulation.

The shader is small:, store this in `prefix_compute.wgsl`:

```wgsl
@group(0) @binding(0) var<storage, read_write> binSizes : array<u32>;
@group(0) @binding(1) var<storage, read_write> prefixes : array<u32>;

@compute @workgroup_size(1,1,1)
fn cs(@builtin(global_invocation_id) cell : vec3u) {
  prefixes[0] = 0;
  for( var i:u32 = 1; i <= arrayLength(&binSizes); i++ ) {
    prefixes[i] = prefixes[i-1] + binSizes[i-1]; 
  }
}
```

Basically we're just keeping a running total of the binSizes and storing the result. Note that we do begin the prefix sums at index 1 instead of zero due to [how the underlying algorithm works](https://www.reddit.com/r/leetcode/comments/1ewxiqt/a_visual_guide_to_prefix_sums/) . That means when we create our `prefixes` array we'll need to make it one larger than our array of binSizes. 

```js
const prefix_shader = await seagulls.import( './prefix_compute.wgsl' )
const prefix_b = sg.buffer( new Float32Array(GRID_SIZE*GRID_SIZE + 1), '', usage )
const prefix = sg.compute({
  shader: prefix_shader,
  data:[
    sizes_b,
    prefix_b
  ],
  dispatchCount:[1,1,1]
})
```

Note that we're using a dispatch count of [1,1,1] and also using [1,1,1] for our workgroup size. The for loop in the shader will only run a single time.  We'll use the results both to organize our boids into memory and when we're doing the simulation itself.

### Re-ordering the memory
OK, so now we know the starting index for each bin. The next stage is:

1. Loop through each boid in our input particles buffer (we'll pingpong this)
2. Calculate what bin the boid is
3. Look up the starting index for this bin
4. Add the number of boids we've already placed into this bins memory location and increment this number
5. Place the boid at the calculated index in our output particles buffer

This stage isn't really any trickier than our initial bin size calculation. Here's the shsader, name it `place_compute.wgsl`:

```wgsl
struct Particle {
  pos: vec2f,
  vel: vec2f
};

const GRID_SIZE:i32 = 60;

@group(0) @binding(0) var<storage> particles_in  : array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out : array<Particle>;
@group(0) @binding(2) var<storage, read_write> binSizes : array<u32>;

// current count (per-invocation) of bin, which we increment when we place particle
@group(0) @binding(3) var<storage, read_write> binCurrentCount : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> prefixes : array<u32>;

@compute @workgroup_size(64,1,1)
fn cs(@builtin(global_invocation_id) id : vec3u) {
  if (id.x >= arrayLength(&particles_in)) { return; }

  // read the particle data
  let particle = particles_in[id.x];

  // compute the linearized bin index
  let binIndex = getBinIndex( particle.pos );

  // reduce bin sizes up to bin index
  var indexSum: u32 = prefixes[ binIndex + 1 ];

  // atomic add to current bin size
  let outputIndex = atomicAdd(&binCurrentCount[binIndex], 1u) + indexSum;

  // assign to output particle array
  particles_out[ outputIndex ] = particle;
}

fn getBinIndex(position : vec2f) -> i32 {
  // position is -1,1, offset by one and scale by half the bin size
  let binxy = vec2i( 
    i32( (1+position.x) * f32(GRID_SIZE/2) ),
    i32( (1+position.y) * f32(GRID_SIZE/2) ) 
  );

  return binxy.y * GRID_SIZE + binxy.x;
}

```

We'll need to create  new buffer to store the current count of each bin as we place our particles. The rest of the data has already been created. In main.js add the following:

```js
const count_b = sg.buffer( new Float32Array( GRID_SIZE*GRID_SIZE ) )
const place = sg.compute({
  shader: place_particles_shader + getBinIndex,
  data:[
    sg.pingpong(state_b2,state_b),
    sizes_b,
    count_b,
    prefix_b
  ],

  dispatchCount:[NUM_PARTICLES / (WORKGROUP_SIZE * WORKGROUP_SIZE),1,1]
})
```

Go ahead and test the full shader pipeline to make sure there aren't any bugs at this point:

`await sg.once( sizes, prefix, place, compute, render )`

Assuming that works, we're all setup to edit our final compute shader that runs the simluation!

### Final Simulation

OK, this is a relatively long shader, but much of it looks the same as our originaal compute shader for boids. Here's what's changing:

1. We'll create a function, `processBin`, that will perform all the simulatino calculations for a given boid and a bin of interest. 
2. We'll run this function for five bins... the bin each boid occupies and its NSEW neighbors. 
3. We'll have to do a little bit of math to calculate the bin indices for each of these neighbors, but it's mostly just addition.
4. In the event that a boid has 0 neighbors, we need to make sure we don't accidentally divide by zero when performing our calculations

```wgsl
struct Particle {
  pos: vec2f,
  vel: vec2f
};

const SIZE:f32 = 60.;

@group(0) @binding(0) var<uniform> res:   vec2f;
@group(0) @binding(1) var<storage> state_r: array<Particle>;
@group(0) @binding(2) var<storage, read_write> state_w: array<Particle>;
@group(0) @binding(3) var<storage, read_write> binSizes : array<u32>;
@group(0) @binding(4) var<storage, read_write> prefixes : array<u32>;

fn processBin(
  boid: Particle,
  boididx:    u32,
  boidBinIdx: u32,
  binStartingIndex: u32,
  center:   ptr<function,vec2f>, 
  keepaway: ptr<function,vec2f>, 
  vel:      ptr<function,vec2f>,
) -> u32 {
  var count = 0u;
  let binSize = binSizes[ boidBinIdx ];
  // hard limit
  let loopSize = select( binSize, 1024, binSize > 1024 );

  for( var i:u32 = 0; i < loopSize; i++ ) {
    // don't use boids' own properties in calculations
    if( boididx == i+binStartingIndex ) { continue; }

    let _boid = state_r[ binStartingIndex + i ];

    // rule 1
    *center += _boid.pos;
    
    // rule 2
    //if( length( _boid.pos - boid.pos ) < .15 ) {
      *keepaway -= ( _boid.pos - boid.pos );
    //}
   
    // rule 3
    *vel += _boid.vel;
    
    count++;
  }
  return count;
}


@compute
@workgroup_size(64,1)

fn cs(@builtin(global_invocation_id) cell:vec3u)  {
  let idx            = cell.x;
  if( idx > arrayLength(&state_r) ){ return; }
  
  var count: u32 = 0;
  var boid:Particle  = state_r[ idx ];
  
  var topidx = getBinIndex( boid.pos ); // bin above boid
  var i:i32 = 0; // keep track of index
    
  var center:vec2f   = vec2f(0.); // rule 1
  var keepaway:vec2f = vec2f(0.); // rule 2
  var vel:vec2f      = vec2f(0.); // rule 3

  count += processBin( boid, cell.x, u32(topidx), 0, &center, &keepaway, &vel ); // top
  i += i32(SIZE) - 1;
  count += processBin( boid, cell.x, u32(i), prefixes[i+1], &center, &keepaway, &vel ); // left
  i++;
  count += processBin( boid, cell.x, u32(i), prefixes[i+1], &center, &keepaway, &vel ); // center
  i++;
  count += processBin( boid, cell.x, u32(i), prefixes[i+1], &center, &keepaway, &vel ); // right
  i += i32(SIZE) - 1;
  count += processBin( boid, cell.x, u32(i), prefixes[i+1], &center, &keepaway, &vel ); // bottom 


  // apply effects of rule 1
  center /= f32( count - 1u );
  boid.vel += (center-boid.pos) * .05;

  // apply effects of rule 2
  boid.vel += keepaway;

  // apply effects of rule 3
  vel /= f32( count - 1u );
  boid.vel += vel * .01;

  // limit speed
  if( length( boid.vel ) > 5. ) {
    boid.vel = (boid.vel / length(boid.vel)) * 5.;
  }
  
  // calculate next position
  boid.pos = boid.pos + (2. / res) * boid.vel;

  // boundaries
  if( abs( boid.pos.y ) >= 1. ) { 
    boid.vel.y *= -1.; 
  }
  if( abs( boid.pos.x ) > 1. ) {
    boid.vel.x *= -1.;
  }

  state_w[ idx ] = boid;
}

fn getBinIndex(position : vec2f) -> i32 {
  // position is -1,1, offset by one and scale by half the bin size
  let binxy = vec2i( 
    i32( (1+position.x) * SIZE/2 ),
    i32( (1+position.y) * SIZE/2 ) 
  );

  return binxy.y * i32(SIZE) + binxy.x;
}

```

OK, least but not least let's update our javascript:

```js
const compute = sg.compute({
  shader: compute_shader + getBinIndex,
  data:[
    res_u,
    sg.pingpong(state_b2, state_b),
    sizes_b,
    prefix_b
  ],
  dispatchCount:[NUM_PARTICLES / (WORKGROUP_SIZE * WORKGROUP_SIZE),1,1]
})
```

Our call to `sg.run()` should look as follows:

`sg.run( sizes, prefix, place, compute, render )`

If it runs, try updating the value of `NUM_PARTICLES` to be something large... if you have a discrete GPU a million particles seems possible, on my integrated gpu I can do around 200,000. This all depends on the size of your grid... higher values for the grid size will let you use more agents, but they'll be less influenced by their neighbors. It's a tradeoff.

## Cleanup
This tutorial does some annoying things. First, you have to enter a variable for the grid size in all the different shaders; this means if you want to change the grid size (which is fun to experiment with!) you have to change it in multiple places. In a similar vein, the `getBinIndex()` function is found in multiple shaders, which means changes to it with have to be implemented in multiple spots. 

You can fix this just by adding these strings to your shaders. For example, if we did something like:

```js
const size_str = `const  SIZE:f32 = 60.;\n` 
```

we could then add it to  any shader just using, for example:

```js
const prefix_shader = size_str + await seagulls.import( './prefix_compute.wgsl' )
```

Now we would only have to edit that SIZE variable in a single location. You can add the `getBinIndex` function to your shaders in the same way.
