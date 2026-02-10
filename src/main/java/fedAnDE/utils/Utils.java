package fedAnDE.utils;

import weka.classifiers.bayes.net.estimate.DiscreteEstimatorBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.estimators.Estimator;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class Utils {

    private static Random random = new Random();

    public static String readFile(String path) {
        String file = null;
        try {
            file = Files.readString(Path.of(path), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return file;
    }

    /**
     * Gets the probability table of an array of WEKA estimators
     *
     * @param estimators Array of WEKA estimators
     * @return The probability table of the estimators
     */
    public static double[][] getProbabilityTable(Estimator[] estimators) {
        int nRows = estimators.length;
        int nCols = ((DiscreteEstimatorBayes) estimators[0]).getNumSymbols();
        double[][] table = new double[nRows][nCols];

        for (int i = 0; i < nRows; i++) {
            Estimator estimator = estimators[i];
            for (int j = 0; j < nCols; j++) {
                table[i][j] = estimator.getProbability(j);
            }
        }
        return table;
    }

    /**
     * Generates all combinations of k indices from 0 to n-1.
     */
    public static List<int[]> generateCombinations(int n, int k) {
        List<int[]> result = new ArrayList<>();
        if (k == 0) {
            result.add(new int[0]);
            return result;
        }

        int[] comb = new int[k];
        for (int i = 0; i < k; i++)
            comb[i] = i;

        while (comb[0] <= n - k) {
            result.add(comb.clone());

            int t = k - 1;
            while (t != 0 && comb[t] == n - k + t)
                t--;
            comb[t]++;
            for (int i = t + 1; i < k; i++) {
                comb[i] = comb[i - 1] + 1;
            }
        }

        return result;
    }

    /**
     * Redefines the class attribute of a dataset by combining selected attributes
     * and the original class,
     * using a predefined mapping to ensure consistency across distributed clients.
     *
     * @param original       The original dataset.
     * @param indices        The indices of the attributes to combine with the
     *                       original class.
     * @param globalClassMap A predefined global map of synthetic class labels to
     *                       index.
     * @return A new Instances object with the synthetic class as its new class
     *         attribute.
     */
    public static Instances redefineClassAttribute(Instances original, int[] indices,
            Map<String, Integer> globalClassMap) {
        // 1. Create a deep copy of the original dataset to avoid modifying it
        Instances data = new Instances(original);

        // 2. Compute the joint value of selected attributes + original class for each
        // instance
        int n = data.numInstances();
        String[] syntheticLabels = new String[n];
        for (int i = 0; i < n; i++) {
            Instance inst = data.instance(i);
            StringBuilder sb = new StringBuilder();
            for (int idx : indices) {
                sb.append(inst.stringValue(idx)).append("|||");
            }
            sb.append(inst.stringValue(data.classIndex()));
            syntheticLabels[i] = sb.toString();
        }

        // 3. Remember the old class index
        int oldClassIndex = data.classIndex();

        // 4. Create the new class attribute using the globally shared list of class
        // values
        List<String> classValuesOrdered = new ArrayList<>(globalClassMap.keySet());
        Attribute newClassAttr = new Attribute("AnDE_Class", classValuesOrdered);
        data.insertAttributeAt(newClassAttr, data.numAttributes());
        data.setClassIndex(data.numAttributes() - 1); // Temporarily set new class index

        // 5. Assign each instance its new class value
        for (int i = 0; i < n; i++) {
            data.instance(i).setValue(data.classIndex(), globalClassMap.get(syntheticLabels[i]));
        }

        // 6. Remove the original class attribute (no longer needed)
        data.deleteAttributeAt(oldClassIndex);

        // 7. Reset the class index to the new last attribute
        data.setClassIndex(data.numAttributes() - 1);

        return data;
    }

    /**
     * Ensures that all class values are present in the dataset, even if they are
     * not in the original data.
     * 
     * @param data     The dataset to check.
     * @param classMap A map of class values to their corresponding indices.
     * @return The modified dataset with all class values present.
     */
    public static Instances ensureAllClassValuesPresent(Instances data, Map<String, Integer> classMap) {
        Instances copy = new Instances(data);
        Set<String> existing = new HashSet<>();

        for (int i = 0; i < copy.numInstances(); i++) {
            String label = copy.instance(i).stringValue(copy.classIndex());
            existing.add(label);
        }

        for (String expected : classMap.keySet()) {
            if (!existing.contains(expected)) {
                // Create dummy instance with missing attributes but correct class label
                double[] vals = new double[copy.numAttributes()];
                Arrays.fill(vals, weka.core.Utils.missingValue());
                vals[copy.classIndex()] = copy.classAttribute().indexOfValue(expected);
                copy.add(new DenseInstance(1.0, vals));
            }
        }

        return copy;
    }

    /**
     * Extracts the class map from the modified dataset.
     *
     * @param modified The modified dataset.
     * @return A map of class values to their corresponding indices.
     */
    public static Map<String, Integer> extractClassMap(Instances modified) {
        Map<String, Integer> classMap = new LinkedHashMap<>();
        Attribute classAttr = modified.classAttribute();
        Enumeration<Object> values = classAttr.enumerateValues();
        int i = 0;
        while (values.hasMoreElements()) {
            classMap.put((String) values.nextElement(), i++);
        }
        return classMap;
    }
}
