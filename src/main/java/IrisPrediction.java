import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class IrisPrediction {
    public static void main(String[] args) throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("irisModel.zip");
        String labels[]={"iris-setosa","iris-versicolor","iris-virginica"};
        System.out.println("---------------------------");
        System.out.println("Prédiction");
        INDArray inputData = Nd4j.create(new double[]{5.1,3.5,1.4,0.2,5.1,3.5,1.4,1.4,4.3,3.5,3,2,6,4,1.7,1},new int[]{4,4});
        INDArray output = model.output(inputData);
        int[] classes = output.argMax(1).toIntVector();
        System.out.println(output);
        for(int i=0 ; i<classes.length ; i++){
            System.out.println("Classe : "+labels[classes[i]]);

        }
    }
}
