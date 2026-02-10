package fedAnDE.privacy;

import java.util.Locale;
import java.util.Random;

/**
 * Simulates the Randomized Response mechanism for ε-differential privacy.
 * This class runs a series of trials for different epsilon values to demonstrate
 * the trade-off between privacy (lower epsilon) and data fidelity.
 */
public class RResponseSimulator {

    /**
     * A simple record to hold the results of a simulation run.
     * @param trueCount The number of times the noisy response matched the original.
     * @param falseCount The number of times the noisy response was flipped.
     */
    private record SimulationResult(int trueCount, int falseCount) {}

    public static void main(String[] args) {
        // Set locale to US to ensure '.' is used as the decimal separator in the output.
        Locale.setDefault(Locale.US);

        // Epsilon values to test, matching the ones from your example.
        double[] epsilons = {
                0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 100.0
        };

        // Number of trials for each epsilon value to get stable empirical counts.
        int numTrials = 100000;

        // A single random number generator for all simulations.
        Random rng = new Random();

        // Loop through each epsilon value and run the simulation.
        for (double epsilon : epsilons) {
            // --- 1. Calculate Theoretical Probabilities ---

            double p_truth;
            double q_lie;

            // Handle very large epsilon values to prevent floating-point overflow (Math.exp(100) -> Infinity).
            // For epsilon > ~37, the probability of telling the truth is effectively 1.0.
            if (epsilon > 37) {
                p_truth = 1.0;
                q_lie = 0.0;
            } else {
                double expEpsilon = Math.exp(epsilon);
                p_truth = expEpsilon / (expEpsilon + 1.0);
                q_lie = 1.0 / (expEpsilon + 1.0);
            }

            // --- 2. Run the Simulation ---

            int trueCount = 0;
            // We assume the original, private value is always 'true' for this simulation.
            // The logic would be the same if it were always 'false'.
            for (int i = 0; i < numTrials; i++) {
                // Generate a random number between 0.0 and 1.0
                if (rng.nextDouble() < p_truth) {
                    // With probability p_truth, the response is the same as the original.
                    trueCount++;
                }
                // Otherwise, with probability q_lie, the response is flipped (counted as a false response).
            }

            SimulationResult result = new SimulationResult(trueCount, numTrials - trueCount);

            // --- 3. Print the Results ---

            // Print theoretical probabilities with high precision.
            System.out.printf(
                    "Randomized Response with epsilon: %.1f, p_truth: %.14f, q_lie: %s%n",
                    epsilon,
                    p_truth,
                    // Use scientific notation for very small q_lie values.
                    String.format(q_lie < 1E-4 && q_lie > 0 ? "%.15E" : "%.14f", q_lie)
            );

            // Print empirical counts from the simulation.
            System.out.printf(
                    "Epsilon: %.1f, True count: %d, False count: %d%n%n",
                    epsilon,
                    result.trueCount(),
                    result.falseCount()
            );
        }
    }
}