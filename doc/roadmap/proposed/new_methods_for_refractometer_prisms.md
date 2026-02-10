# need to add some methods to build up a refractometerPrism

vertex layout (before rotation):
        M
    V3---------V2
     \          /
   E  \        / X   ← Edge 2 (V2→V3 is NOT an edge;
       \      /        edges go 0→1→2→3→0)
       V0----V1
          B

Edge 0: Base (B) | facing South Edge (S)
Edge 1: Exit Face (X) | facing East Edge (E)
Edge 2: Measuring Surface (M) | facing North Edge (N)
Edge 3: Entrance Face (E) | facing West Edge (W)'

The peculiarity of this geometry is that the lenght of the Measuring Surface and the height of the prism are completely calculated by the angle of the total internal reflection.
If we call 