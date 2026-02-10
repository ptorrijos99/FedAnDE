package fedAnDE.experiments;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class to run experiments.
 * Separates the argument parsing and configuration logic from the experiment
 * class.
 */
public class ExperimentRunner {

    public static void main(String[] args) {
        // Default dataset and experimental configuration
        String nodeName = "localhost";

        String folder = "Discrete";
        String datasetName = "Car_Evaluation";
        int nClients = 20;
        int seed = 42;
        int nFolds = 2;
        int nIterations = 1;

        // Structure and parameter learning configurations
        String structure = "NB"; // Possible values: "NB", "A1DE", "A2DE", ..., "AnDE"
        String parameterLearning = "wCCBN"; // Possible values: "dCCBN", "wCCBN", "eCCBN", and "Weka"
        String maxIterations = "1";

        // Fusion behaviour
        boolean fuseParameters = true;
        boolean fuseProbabilities = true;
        int nBins = -1;

        // IID or no-IID
        double alpha = 0; // IID if <=0

        // Differential privacy parameters
        String dpType = "Laplace"; // "Laplace", "Gaussian", "ZCDP", "None"
        double epsilon = 1000000; // for Laplace and Gaussian
        double delta = 1e-5; // only for Gaussian
        double rho = 0.1; // only for ZCDP
        double sensitivity = 1.0; // default for PT; smaller for WDPT
        boolean autoSensitivity = true; // Calculate sensitivity automatically based on the data columns

        // Check if arguments are provided
        if (args.length > 0) {
            int index = Integer.parseInt(args[0]);
            String paramsFileName = args[1];
            nodeName = args[2];

            // Read the parameters from args
            String[] parameters = null;
            try (BufferedReader br = new BufferedReader(new FileReader(paramsFileName))) {
                String line;
                for (int i = 0; i < index; i++)
                    br.readLine();
                line = br.readLine();
                parameters = line.split(" ");
            } catch (Exception e) {
                System.out.println(e);
                return;
            }

            System.out.println("Number of hyperparams: " + parameters.length);
            int i = 0;
            for (String string : parameters) {
                System.out.println("Param[" + i + "]: " + string);
                i++;
            }

            // Read the parameters from file
            folder = parameters[0];
            datasetName = parameters[1];
            nClients = Integer.parseInt(parameters[2]);
            seed = Integer.parseInt(parameters[3]);
            nFolds = Integer.parseInt(parameters[4]);
            nIterations = Integer.parseInt(parameters[5]);
            structure = parameters[6];
            parameterLearning = parameters[7];
            maxIterations = parameters[8];
            fuseParameters = Boolean.parseBoolean(parameters[9]);
            fuseProbabilities = Boolean.parseBoolean(parameters[10]);
            nBins = Integer.parseInt(parameters[11]);

            if (parameters.length > 12) {
                try {
                    alpha = Double.parseDouble(parameters[12]);
                } catch (NumberFormatException e) {
                    System.out.println("Warning: Could not parse Alpha at index 12. Defaulting to -1.");
                    alpha = -1.0;
                }
            }

            // Add DP parameters if passed
            if (parameters.length >= 16) {
                dpType = parameters[13];
                sensitivity = Double.parseDouble(parameters[14]);
                autoSensitivity = Boolean.parseBoolean(parameters[15]);

                if (dpType.equalsIgnoreCase("Laplace")) {
                    epsilon = Double.parseDouble(parameters[16]);
                } else if (dpType.equalsIgnoreCase("Gaussian")) {
                    epsilon = Double.parseDouble(parameters[16]);
                } else if (dpType.equalsIgnoreCase("ZCDP")) {
                    rho = Double.parseDouble(parameters[16]);
                }

                if (dpType.equalsIgnoreCase("Gaussian")) {
                    delta = Double.parseDouble(parameters[17]);
                }
            } else {
                dpType = "None";
            }
        }

        // Use supervised discretization in case the number of bins is not provided and
        // equal-frequency otherwise
        String[] discretizerOptions = nBins == -1 ? new String[] {} : new String[] { "-F", "-B", "" + nBins };

        // Options for local learning algorithm
        String[] algorithmOptions = new String[] { "-S", structure, "-P", parameterLearning, "-I", maxIterations };

        // Build shared fusion flags based on configuration
        List<String> flags = new ArrayList<>();
        if (fuseParameters)
            flags.add("-FP"); // Fuse parameter vectors
        if (fuseProbabilities)
            flags.add("-FPR"); // Fuse class-conditional probabilities

        // Convert flags to arrays for client and server options
        String[] type = new String[0];
        String[] clientOptions = flags.toArray(type);
        String[] serverOptions = flags.toArray(type);

        // Create DP options
        List<String> dpList = new ArrayList<>();
        switch (dpType.toLowerCase()) {
            case "laplace" -> dpList.addAll(List.of("-DP", "Laplace", "-E", "" + epsilon, "-S", "" + sensitivity));
            case "gaussian" ->
                dpList.addAll(List.of("-DP", "Gaussian", "-E", "" + epsilon, "-D", "" + delta, "-S", "" + sensitivity));
            case "zcdp" -> dpList.addAll(List.of("-DP", "ZCDP", "-R", "" + rho, "-S", "" + sensitivity));
        }
        if (autoSensitivity)
            dpList.add("-AUTO");
        String[] dpOptions = dpList.toArray(new String[0]);

        // Create output suffix for result identification
        String suffix = datasetName + "_" + nBins + "_" + structure + "_" + parameterLearning + "_" + maxIterations
                + "_" + fuseParameters + "_" + fuseProbabilities
                + "_" + dpType + "_" + epsilon + "_" + delta + "_" + rho + "_" + sensitivity + "_" + autoSensitivity
                + "_" + nClients + "_" + seed + "_" + nIterations + "_" + nFolds + "_a" + alpha + ".csv";

        // Run the experiment
        CCBNExperiment.run(folder, datasetName, discretizerOptions, algorithmOptions, clientOptions, serverOptions,
                dpOptions, nClients, nIterations, nFolds, seed, alpha, suffix, nodeName);
    }
}
