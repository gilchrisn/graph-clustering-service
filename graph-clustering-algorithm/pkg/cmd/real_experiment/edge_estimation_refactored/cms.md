A **Count-Min Sketch (CMS)** is a probabilistic data structure used to estimate the frequency of items in a large stream of data. It's highly memory-efficient, as it avoids storing exact counts for every item.

### Structure and Operations

As implemented in your Go code, a CMS consists of a 2D matrix of counters with dimensions `d` (depth) x `w` (width), along with `d` independent hash functions.

* **Add (Update)**: When an item is added to the sketch, it is hashed by all `d` hash functions. Each hash function maps the item to a specific bucket in its corresponding row, and that bucket's counter is incremented by one.
* **Estimate (Query)**: To estimate an item's frequency, the item is again hashed by all `d` functions to find its `d` corresponding buckets. The estimated frequency is the **minimum** value found among those `d` counters. The minimum is chosen because it corresponds to the row with the fewest random collisions, making it the most accurate estimate.

### Key Properties

The primary feature of a CMS is its **one-sided error guarantee**:
* The estimate is **always an overestimate** or exactly correct, but never an underestimate ($\hat{f}_x \ge f_x$).
* This overestimation is caused by **hash collisions**, where different items are mapped to the same counter, inflating its value.
* The accuracy is controlled by the sketch's dimensions. Increasing the **width (`w`)** reduces the probability of collisions, while increasing the **depth (`d`)** increases the probability of finding a bucket with less collision noise.
