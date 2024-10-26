"""
Part 2: Performance Comparisons

**Released: Wednesday, October 16**

In this part, we will explore comparing the performance
of different pipelines.
First, we will set up some helper classes.
Then we will do a few comparisons
between two or more versions of a pipeline
to report which one is faster.
"""

import part1
import time
from collections import defaultdict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pandasql as ps
import matplotlib.ticker as mticker

"""
=== Questions 1-5: Throughput and Latency Helpers ===

We will design and fill out two helper classes.

The first is a helper class for throughput (Q1).
The class is created by adding a series of pipelines
(via .add_pipeline(name, size, func))
where name is a title describing the pipeline,
size is the number of elements in the input dataset for the pipeline,
and func is a function that can be run on zero arguments
which runs the pipeline (like def f()).

The second is a similar helper class for latency (Q3).

1. Throughput helper class

Fill in the add_pipeline, eval_throughput, and generate_plot functions below.
"""

# Number of times to run each pipeline in the following results.
# You may modify this if any of your tests are running particularly slow
# or fast (though it should be at least 10).
NUM_RUNS = 10

class ThroughputHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline input sizes
        self.sizes = []

        # Pipeline throughputs
        # This is set to None, but will be set to a list after throughputs
        # are calculated.
        self.throughputs = None

    def add_pipeline(self, name, size, func):
        self.names.append(name)
        self.sizes.append(size)
        self.pipelines.append(func)

    def compare_throughput(self):
        # Measure the throughput of all pipelines
        # and store it in a list in self.throughputs.
        # Make sure to use the NUM_RUNS variable.
        # Also, return the resulting list of throughputs,
        # in **number of items per second.**
        
        throughputs_avgs = []

        for i, pipeline in enumerate(self.pipelines):
            total_time = 0.0

            for _ in range(NUM_RUNS):
                start_time = time.perf_counter()
                pipeline()
                end_time = time.perf_counter()
                total_time +=  (end_time - start_time)

            avg_time = total_time / NUM_RUNS

            size = self.sizes[i]
            print("\n\n")
            print("\n\n")
            print(size)
            print(avg_time)
            print("\n\n")
            print("\n\n")
            throughput = size / avg_time if avg_time > 0 else float('inf')
            throughputs_avgs.append(throughput)

        self.throughputs = throughputs_avgs
        return throughputs_avgs

    def generate_plot(self, filename):
        # Generate a plot for throughput using matplotlib.
        # You can use any plot you like, but a bar chart probably makes
        # the most sense.
        # Make sure you include a legend.
        # Save the result in the filename provided.
        plt.close()
        plt.bar(x = self.names, height = self.throughputs)
        plt.title("Throughputs of various pipelines in ops/second")
        # Add labels and title
        plt.xlabel('Pipeline Names')
        plt.ylabel('Throughput (million items per second)')
        plt.title('Pipeline Throughput Comparison')
        plt.tight_layout()
        plt.savefig(filename)

"""
As your answer to this part,
return the name of the method you decided to use in
matplotlib.

(Example: "boxplot" or "scatter")
"""

def q1():
    # Return plot method (as a string) from matplotlib
    return "bar"

"""
2. A simple test case

To make sure your monitor is working, test it on a very simple
pipeline that adds up the total of all elements in a list.

We will compare three versions of the pipeline depending on the
input size.
"""

LIST_SMALL = [10] * 100
LIST_MEDIUM = [10] * 100_000
LIST_LARGE = [10] * 100_000_000

def add_list(l):
    # TODO
    total = 0
    for i in l:
        total += i

    return total

def q2a():
    # Create a ThroughputHelper object
    h = ThroughputHelper()
    # Add the 3 pipelines.
    # (You will need to create a pipeline for each one.)
    # Pipeline names: small, medium, large
    h.add_pipeline("small", len(LIST_SMALL), lambda: add_list(LIST_SMALL))
    h.add_pipeline("medium", len(LIST_MEDIUM), lambda: add_list(LIST_MEDIUM))
    h.add_pipeline("large", len(LIST_LARGE), lambda: add_list(LIST_LARGE))

    throughputs = h.compare_throughput()
    # Generate a plot.
    # Save the plot as 'output/q2a.png'.
    h.generate_plot("output/q2a.png")
    # Finally, return the throughputs as a list.
    return throughputs

"""
2b.
Which pipeline has the highest throughput?
Is this what you expected?

=== ANSWER Q2b BELOW ===

It looks like the medium and large have the highest throughput.
This is what I expected, as the overhead time for small
probably decreases its throughput, but is heavily amortized
in the medium and large runs. 

=== END OF Q2b ANSWER ===
"""

"""
3. Latency helper class.

Now we will create a similar helper class for latency.

The helper should assume a pipeline that only has *one* element
in the input dataset.

It should use the NUM_RUNS variable as with throughput.
"""

class LatencyHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline latencies
        # This is set to None, but will be set to a list after latencies
        # are calculated.
        self.latencies = None

    def add_pipeline(self, name, func):
        self.names.append(name)
        self.pipelines.append(func)


    def compare_latency(self):
        # Measure the latency of all pipelines
        # and store it in a list in self.latencies.
        # Also, return the resulting list of latencies,
        # in **milliseconds.**

        latencies = []

        for pipeline in self.pipelines:
            total_time = 0.0

            for _ in range(NUM_RUNS):
                start_time = time.perf_counter()
                pipeline()
                end_time = time.perf_counter()
                total_time += (end_time - start_time)
            
            lat = (total_time / NUM_RUNS) * 1000  # Convert seconds to milliseconds
            latencies.append(lat)
        
        self.latencies = latencies
        return latencies
        

    def generate_plot(self, filename):
        # Generate a plot for latency using matplotlib.
        # You can use any plot you like, but a bar chart probably makes
        # the most sense.
        # Make sure you include a legend.
        # Save the result in the filename provided.
        plt.close()
        plt.bar(x = self.names, height = self.latencies)
        plt.title("Latencies of various pipelines in ms")
        # Add labels and title
        plt.xlabel('Pipeline Names')
        plt.ylabel('Latencies (milliseconds second)')
        plt.title('Pipeline Latency Comparison')
        plt.tight_layout()
        plt.savefig(filename)

"""
As your answer to this part,
return the number of input items that each pipeline should
process if the class is used correctly.
"""

def q3():
    # Return the number of input items in each dataset,
    # for the latency helper to run correctly.
    return 1

"""
4. To make sure your monitor is working, test it on
the simple pipeline from Q2.

For latency, all three pipelines would only process
one item. Therefore instead of using
LIST_SMALL, LIST_MEDIUM, and LIST_LARGE,
for this question run the same pipeline three times
on a single list item.
"""

LIST_SINGLE_ITEM = [10] # Note: a list with only 1 item

def q4a():
    # Create a LatencyHelper object
    h = LatencyHelper()
    # Add the single pipeline three times.
    h.add_pipeline("pipeline 1", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("pipeline 2", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("pipeline 3", lambda: add_list(LIST_SINGLE_ITEM))
    # Generate a plot.
    latencies = h.compare_latency()
    # Save the plot as 'output/q4a.png'.
    h.generate_plot("output/q4a.png")
    # Finally, return the latencies as a list.
    return latencies

"""
4b.
How much did the latency vary between the three copies of the pipeline?
Is this more or less than what you expected?

=== ANSWER Q1b BELOW ===

It's actually pretty interesting how much the latency decreased
between runs of the pipeline. I would guess this is related to
how Python manages memory and processes under the hood. 

=== END OF Q1b ANSWER ===
"""

"""
Now that we have our helpers, let's do a simple comparison.

NOTE: you may add other helper functions that you may find useful
as you go through this file.

5. Comparison on Part 1

Finally, use the helpers above to calculate the throughput and latency
of the pipeline in part 1.
"""

# You will need these:
# part1.load_input()
# part1.PART_1_PIPELINE()

part1_dataframes = part1.load_input()

def q5a():
    # Return the throughput of the pipeline in part 1.
    h = ThroughputHelper()
    input_size = sum(len(df) for df in part1_dataframes)
    h.add_pipeline("pipeline 1",input_size, lambda: part1.PART_1_PIPELINE)
    lat = h.compare_throughput()

    return lat

def q5b():
    # Return the latency of the pipeline in part 1.
    h = LatencyHelper()
    h.add_pipeline("pipeline 1", lambda: part1.PART_1_PIPELINE)
    lat = h.compare_latency()
    return lat

"""
===== Questions 6-10: Performance Comparison 1 =====

For our first performance comparison,
let's look at the cost of getting input from a file, vs. in an existing DataFrame.

6. We will use the same population dataset
that we used in lecture 3.

Load the data using load_input() given the file name.

- Make sure that you clean the data by removing
  continents and world data!
  (World data is listed under OWID_WRL)

Then, set up a simple pipeline that computes summary statistics
for the following:

- *Year over year increase* in population, per country

    (min, median, max, mean, and standard deviation)

How you should compute this:

- For each country, we need the maximum year and the minimum year
in the data. We should divide the population difference
over this time by the length of the time period.

- Make sure you throw out the cases where there is only one year
(if any).

- We should at this point have one data point per country.

- Finally, as your answer, return a list of the:
    min, median, max, mean, and standard deviation
  of the data.

Hints:
You can use the describe() function in Pandas to get these statistics.
You should be able to do something like
df.describe().loc["min"]["colum_name"]

to get a specific value from the describe() function.

You shouldn't use any for loops.
See if you can compute this using Pandas functions only.
"""

def load_input(filename):
    # Return a dataframe containing the population data
    # **Clean the data here**
    population_data = pd.read_csv(filename)
    # groups = population.groupby("Code").mean("Population (historical)")

    filtered_data = population_data[population_data["Code"].notnull()]
    return filtered_data

def population_pipeline(df):
    # Input: the dataframe from load_input()
    # Return a list of min, median, max, mean, and standard deviation
    population_data = df
    population_data = population_data[population_data["Code"].notnull()]

    country_stats = population_data.groupby("Entity").apply(lambda group: pd.Series({
        "min_year": group["Year"].min(),
        "max_year": group["Year"].max(),
        "min_population": group.loc[group["Year"] == group["Year"].min(), "Population (historical)"].values[0],
        "max_population": group.loc[group["Year"] == group["Year"].max(), "Population (historical)"].values[0],
    }))

    country_stats['year_range'] = country_stats['max_year'] - country_stats['min_year']
    country_stats['population_growth_per_year'] = (
        (country_stats['max_population'] - country_stats['min_population']) / country_stats['year_range']
    )

    growth_stats = country_stats["population_growth_per_year"].agg(["min", "median", "max", "mean", "std"]).tolist()
    return growth_stats



def q6():
    # As your answer to this part,
    # call load_input() and then population_pipeline()
    # Return a list of min, median, max, mean, and standard deviation
    data = load_input("data/population.csv")
    return population_pipeline(data)
    # raise NotImplementedError

"""
7. Varying the input size

Next we want to set up three different datasets of different sizes.

Create three new files,
    - data/population-small.csv
      with the first 600 rows
    - data/population-medium.csv
      with the first 6000 rows
    - data/population-single-row.csv
      with only the first row
      (for calculating latency)

You can edit the csv file directly to extract the first rows
(remember to also include the header row)
and save a new file.

Make four versions of load input that load your datasets.
(The _large one should use the full population dataset.)
"""

def load_input_small():
    return load_input("data/population.csv").head(600)

def load_input_medium():
    return load_input("data/population.csv").head(6000)

def load_input_large():
    return load_input("data/population.csv")

def load_input_single_row():
    # This is the pipeline we will use for latency.
    # raise NotImplementedError
    return load_input("data/population.csv").head(1)

def q7():
    # Don't modify this part
    s = load_input_small()
    m = load_input_medium()
    l = load_input_large()
    x = load_input_single_row()
    return [len(s), len(m), len(l), len(x)]

"""
8.
Create baseline pipelines

First let's create our baseline pipelines.
Create four pipelines,
    baseline_small
    baseline_medium
    baseline_large
    baseline_latency

based on the three datasets above.
Each should call your population_pipeline from Q7.
"""

def baseline_small():
    h = LatencyHelper()
    h.add_pipeline("baseline small", lambda: population_pipeline(load_input_small()))

def baseline_medium():
    h = LatencyHelper()
    h.add_pipeline("baseline medium", lambda: population_pipeline(load_input_medium()))

def baseline_large():
    h = LatencyHelper()
    h.add_pipeline("baseline large", lambda: population_pipeline(load_input_large()))

def baseline_latency():
    h = LatencyHelper()
    h.add_pipeline("baseline large", lambda: population_pipeline(load_input_single_row()))

def q8():
    # Don't modify this part
    _ = baseline_medium()
    return ["baseline_small", "baseline_medium", "baseline_large", "baseline_latency"]

"""
9.
Finally, let's compare whether loading an input from file is faster or slower
than getting it from an existing Pandas dataframe variable.

Create four new dataframes (constant global variables)
directly in the script.
Then use these to write 3 new pipelines:
    fromvar_small
    fromvar_medium
    fromvar_large
    fromvar_latency

As your answer to this part;
a. Generate a plot in output/q9a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, fromvar_small, fromvar_medium, fromvar_large
b. Generate a plot in output/q9b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, fromvar_latency
"""

# TODO
POPULATION_SMALL = load_input_small()
POPULATION_MEDIUM = load_input_medium()
POPULATION_LARGE = load_input_large()
POPULATION_SINGLE_ROW = load_input_single_row()

def fromvar_small():
    h = ThroughputHelper()
    h.add_pipeline("from var small", 600, lambda: population_pipeline(POPULATION_SMALL))

def fromvar_medium():
    h = ThroughputHelper()
    h.add_pipeline("from var medium", 6000, lambda: population_pipeline(POPULATION_MEDIUM))

def fromvar_large():
    h = ThroughputHelper()
    h.add_pipeline("from var large", 59178, lambda: population_pipeline(POPULATION_LARGE))

def fromvar_latency():
    h = LatencyHelper()
    h.add_pipeline("from var latency", lambda: population_pipeline(POPULATION_SINGLE_ROW))

def q9a():
    h = ThroughputHelper()
    # Add all 6 pipelines for a throughput comparison
    h.add_pipeline("baseline s", 600, lambda: population_pipeline(load_input_small()))
    h.add_pipeline("baseline m", 6000, lambda: population_pipeline(load_input_medium()))
    h.add_pipeline("baseline l", 59178, lambda: population_pipeline(load_input_large()))
    h.add_pipeline("fromvar s", 600, lambda: population_pipeline(POPULATION_SMALL))
    h.add_pipeline("fromvar m", 6000,lambda: population_pipeline(POPULATION_MEDIUM))
    h.add_pipeline("fromvar l", 59178, lambda: population_pipeline(POPULATION_LARGE))
    throughputs = h.compare_throughput()
    # Generate plot in ouptut/q9a.png
    h.generate_plot("output/q9a.png")
    # Return list of 6 throughputs
    return throughputs

def q9b():
    # Add 2 pipelines for a latency comparison
    h = LatencyHelper()
    h.add_pipeline("baseline latency", lambda: population_pipeline(load_input_single_row()))
    h.add_pipeline("from var latency", lambda: population_pipeline(POPULATION_SINGLE_ROW))
    latencies = h.compare_latency()
    # Generate plot in ouptut/q9b.png
    h.generate_plot("output/q9b.png")
    # Return list of 2 latencies
    return latencies

"""
10.
Comment on the plots above!
How dramatic is the difference between the two pipelines?
Which differs more, throughput or latency?
What does this experiment show?

===== ANSWER Q10 BELOW =====

The difference between loading the data from the csv vs a variable
is very dramatic - variable is much faster. The difference between
the fromvar and baseline is greater in the latency department, where
the baseline is almost 7x slower. 

===== END OF Q10 ANSWER =====
"""

"""
===== Questions 11-14: Performance Comparison 2 =====

Our second performance comparison will explore vectorization.

Operations in Pandas use Numpy arrays and vectorization to enable
fast operations.
In particular, they are often much faster than using for loops.

Let's explore whether this is true!

11.
First, we need to set up our pipelines for comparison as before.

We already have the baseline pipelines from Q8,
so let's just set up a comparison pipeline
which uses a for loop to calculate the same statistics.

Create a new pipeline:
- Iterate through the dataframe entries. You can assume they are sorted.
- Manually compute the minimum and maximum year for each country.
- Add all of these to a Python list. Then manually compute the summary
  statistics for the list (min, median, max, mean, and standard deviation).
"""

def for_loop_pipeline(df, multiple_rows = True):
    # Input: the dataframe from load_input()
    # Return a list of min, median, max, mean, and standard deviation

    # 1. Iterate through dataframe to calculate min/max year for each country
    # 2. Group by country
    # 3. Compute growth rate between those years
    # 4. Filter out single year entries
    # 5. Calculate summary statistics on those growth rates

    population_data = df
    population_data = population_data[population_data["Code"].notnull()]

    min_years = defaultdict(lambda: float('inf')) # Algeria: -1000
    min_years_pop = defaultdict(int) # Algeria: 5912
    max_years = defaultdict(lambda: float("-inf")) # Algeria: 2020
    max_years_pop = defaultdict(int) # Algeria: 6019

    # 1. Calculate min/max year for each country
    for index, row in population_data.iterrows():
        country_name = row["Entity"]
        country_year = row["Year"]
        country_population = row["Population (historical)"]

        if country_year < min_years[country_name]:
            min_years[country_name] = country_year
            min_years_pop[country_name] = country_population

        if country_year > max_years[country_name]:
            max_years[country_name] = country_year
            max_years_pop[country_name] = country_population

    deletion_countries = []
    growth_rates = []
    # 3. compute growth rate
    if not multiple_rows:
        for country, year in min_years.items():
            growth_rates.append(max_years_pop[country])

    # 2. Get rid of single year entries
    for country, year in min_years.items():
        if year == max_years[country]:
            deletion_countries.append(country)
    
    for country in deletion_countries:
        del min_years[country]
        del min_years_pop[country]
        del max_years[country]
        del max_years_pop[country]

    
    else:
        for country, year in min_years.items():
            population_diff = max_years_pop[country] - min_years_pop[country]
            year_diff = max_years[country] - min_years[country]
        
            growth_rates.append(population_diff / year_diff)
    
    growth_rates_df = pd.DataFrame(growth_rates)
    return growth_rates_df.agg(['min', 'median', 'max', 'mean', 'std'])

def q11():
    # As your answer to this part, call load_input() and then
    # for_loop_pipeline() to return the 5 numbers.
    # (these should match the numbers you got in Q6.)
    data = load_input('data/population.csv')
    return [float(x[0]) for x in for_loop_pipeline(data).values]

"""
12.
Now, let's create our pipelines for comparison.

As before, write 4 pipelines based on the datasets from Q7.
"""

def for_loop_small():
    h = ThroughputHelper()
    h.add_pipeline("for loop small", 600, lambda: for_loop_pipeline(POPULATION_SMALL))


def for_loop_medium():
    h = ThroughputHelper()
    h.add_pipeline("for loop medium", 6000, lambda: for_loop_pipeline(POPULATION_MEDIUM))


def for_loop_large():
    h = ThroughputHelper()
    h.add_pipeline("for loop large", 59178, lambda: for_loop_pipeline(POPULATION_LARGE))


def for_loop_latency():
    h = LatencyHelper()
    h.add_pipeline("for loop latency", lambda: for_loop_pipeline(POPULATION_SINGLE_ROW, multiple_rows= False))

def q12():
    # Don't modify this part
    _ = for_loop_medium()
    return ["for_loop_small", "for_loop_medium", "for_loop_large", "for_loop_latency"]

"""
13.
Finally, let's compare our two pipelines,
as we did in Q9.

a. Generate a plot in output/q13a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, for_loop_small, for_loop_medium, for_loop_large

b. Generate a plot in output/q13b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, for_loop_latency
"""

def q13a():
    # Add all 6 pipelines for a throughput comparison
    # Generate plot in ouptut/q13a.png
    # Return list of 6 throughputs
    h = ThroughputHelper()
    # Add all 6 pipelines for a throughput comparison
    h.add_pipeline("baseline s", 600, lambda: population_pipeline(POPULATION_SMALL))
    h.add_pipeline("baseline m", 6000, lambda: population_pipeline(POPULATION_MEDIUM))
    h.add_pipeline("baseline l", 59178, lambda: population_pipeline(POPULATION_LARGE))
    h.add_pipeline("for loop s", 600, lambda: for_loop_pipeline(POPULATION_SMALL))
    h.add_pipeline("for loop m", 6000,lambda: for_loop_pipeline(POPULATION_MEDIUM))
    h.add_pipeline("for loop l", 59178, lambda: for_loop_pipeline(POPULATION_LARGE))
    throughputs = h.compare_throughput()
    # Generate plot in ouptut/q9a.png
    h.generate_plot("output/q13a.png")
    # Return list of 6 throughputs
    return throughputs

def q13b():
    # Add 2 pipelines for a latency comparison
    # Generate plot in ouptut/q13b.png
    # Return list of 2 latencies
    h = LatencyHelper()
    h.add_pipeline("baseline latency", lambda: population_pipeline(POPULATION_SINGLE_ROW))
    h.add_pipeline("for loop latency", lambda: for_loop_pipeline(POPULATION_SINGLE_ROW, multiple_rows= False))
    latencies = h.compare_latency()
    h.generate_plot("output/q13b.png")
    return latencies

"""
14.
Comment on the results you got!

14a. Which pipelines is faster in terms of throughput?

===== ANSWER Q14a BELOW =====

In terms of throughput, the Pandas implementations definitely 
blow the throughput of the for loops. 

===== END OF Q14a ANSWER =====

14b. Which pipeline is faster in terms of latency?

===== ANSWER Q14b BELOW =====

However, the for loop latency is much better. Maybe 5x less. 

===== END OF Q14b ANSWER =====

14c. Do you notice any other interesting observations?
What does this experiment show?

===== ANSWER Q14c BELOW =====

It's interesting how the throughput of the pandas 
implementation increases as the size of the dataset
increases, probably due to the overhead of getting
the data ready. However, the for loop's throughput
is always kind of constant. 

===== END OF Q14c ANSWER =====
"""

"""
===== Questions 15-17: Reflection Questions =====
15.

Take a look at all your pipelines above.
Which factor that we tested (file vs. variable, vectorized vs. for loop)
had the biggest impact on performance?

===== ANSWER Q15 BELOW =====

It looks like file vs. variable had the most dramatic performance. 

===== END OF Q15 ANSWER =====

16.
Based on all of your plots, form a hypothesis as to how throughput
varies with the size of the input dataset.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q16 BELOW =====

Throughput generally increases and then plateaus with
the input dataset scaling. We can see this in the pipeline
throughput comparison with pandas vs. the for loop. 
This is only present though if there are optimizations
that can improve the handling of data (the for loop) does
not have them. 

===== END OF Q16 ANSWER =====

17.
Based on all of your plots, form a hypothesis as to how
throughput is related to latency.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q17 BELOW =====

It looks like throughput might actually be
inversely correlated with latency. I have a hypothesis based
on the q13a and q13b plots - Pandas has higher latency
but it has better throughput. 

===== END OF Q17 ANSWER =====
"""

"""
===== Extra Credit =====

This part is optional.

Use your pipeline to compare something else!

Here are some ideas for what to try:
- the cost of random sampling vs. the cost of getting rows from the
  DataFrame manually
- the cost of cloning a DataFrame
- the cost of sorting a DataFrame prior to doing a computation
- the cost of using different encodings (like one-hot encoding)
  and encodings for null values
- the cost of querying via Pandas methods vs querying via SQL
  For this part: you would want to use something like
  pandasql that can run SQL queries on Pandas data frames. See:
  https://stackoverflow.com/a/45866311/2038713

As your answer to this part,
as before, return
a. the list of 6 throughputs
and
b. the list of 2 latencies.

and generate plots for each of these in the following files:
    output/extra_credit_a.png
    output/extra_credit_b.png
"""

# Extra credit (optional)

def population_pipeline_clone_data(data):
    POP_DATA_CLONE = data
    population_pipeline(POP_DATA_CLONE)

def extra_credit_a():
    # Generate plot in ouptut/q13a.png
    # Return list of 6 throughputs
    h = ThroughputHelper()
    h.add_pipeline("no copy s", 600, lambda: population_pipeline(POPULATION_SMALL))
    h.add_pipeline("no copy m", 6000, lambda: population_pipeline(POPULATION_MEDIUM))
    h.add_pipeline("no copy l", 59178, lambda: population_pipeline(POPULATION_LARGE))
    h.add_pipeline("copy df s", 600, lambda: population_pipeline_clone_data(POPULATION_SMALL))
    h.add_pipeline("copy df m", 6000,lambda: population_pipeline_clone_data(POPULATION_MEDIUM))
    h.add_pipeline("copy df l", 59178, lambda: population_pipeline_clone_data(POPULATION_LARGE))
    # Add all 6 pipelines for a throughput comparison
    throughputs = h.compare_throughput()
    h.generate_plot("output/extra_credit_a.png")
    return throughputs

def extra_credit_b():
    h = LatencyHelper()
    h.add_pipeline("no copy", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("copy dataframe", lambda: add_list(LIST_SINGLE_ITEM))
    latencies = h.compare_latency()
    h.generate_plot("output/extra_credit_b.png")
    return latencies

"""
===== Wrapping things up =====

**Don't modify this part.**

To wrap things up, we have collected
your answers and saved them to a file below.
This will be run when you run the code.
"""

ANSWER_FILE = "output/part2-answers.txt"
UNFINISHED = 0

def log_answer(name, func, *args):
    try:
        answer = func(*args)
        print(f"{name} answer: {answer}")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},{answer}\n')
            print(f"Answer saved to {ANSWER_FILE}")
    except NotImplementedError:
        print(f"Warning: {name} not implemented.")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},Not Implemented\n')
        global UNFINISHED
        UNFINISHED += 1

def PART_2_PIPELINE():
    open(ANSWER_FILE, 'w').close()

    # Q1-5
    log_answer("q1", q1)
    log_answer("q2a", q2a)
    # 2b: commentary
    log_answer("q3", q3)
    log_answer("q4a", q4a)
    # 4b: commentary
    log_answer("q5a", q5a)
    log_answer("q5b", q5b)

    # Q6-10
    log_answer("q6", q6)
    log_answer("q7", q7)
    log_answer("q8", q8)
    log_answer("q9a", q9a)
    log_answer("q9b", q9b)
    # 10: commentary

    # Q11-14
    log_answer("q11", q11)
    log_answer("q12", q12)
    log_answer("q13a", q13a)
    log_answer("q13b", q13b)
    # 14: commentary

    # 15-17: reflection
    # 15: commentary
    # 16: commentary
    # 17: commentary

    # Extra credit
    log_answer("extra credit (a)", extra_credit_a)
    log_answer("extra credit (b)", extra_credit_b)

    # Answer: return the number of questions that are not implemented
    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return UNFINISHED

"""
=== END OF PART 2 ===

Main function
"""

if __name__ == '__main__':
    log_answer("PART 2", PART_2_PIPELINE)
