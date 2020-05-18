import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class IrisApp {

    public static void main(String[] args) throws IOException, InterruptedException {
        double learningRate = 0.001;
        int numInputs = 4;
        int numHidden = 10;
        int numOutputs = 3;
        long seed=123;
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHidden)
                        .activation(Activation.SIGMOID)
                        .build())

                .layer(1, new OutputLayer.Builder()
                        .nIn(numHidden)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

        //Démarrage du monitoring
        UIServer uiServer = UIServer.getInstance();
        StatsStorage inMemoryStatsStorage=new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);
        //ScoreIterationListener scoreIterationListener = new ScoreIterationListener(1);
        model.setListeners(new StatsListener(inMemoryStatsStorage));




        System.out.println(configuration.toJson());
        System.out.println("Entrainement du modele");
        File fileTrain = new ClassPathResource("iris-train.csv").getFile();
        RecordReader recordReaderTrain = new CSVRecordReader();
        recordReaderTrain.initialize(new FileSplit(fileTrain));
        int batchSize = 1;
        int labelIndex = 4;
        DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain,batchSize,labelIndex,numOutputs);

       while(dataSetIteratorTrain.hasNext()){
            System.out.println("-----------------");
            DataSet dataSet=dataSetIteratorTrain.next();
            System.out.println(dataSet.getFeatures());
            System.out.println(dataSet.getLabels());

        }


        int nEpochs = 500;
        for(int i=0 ; i<nEpochs;i++){
            model.fit(dataSetIteratorTrain);
        }

        System.out.println("Evaluation du modele");
        File fileTest = new ClassPathResource("iris-test.csv").getFile();
        RecordReader recordReaderTest= new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));
        DataSetIterator dataSetIteratorTest= new RecordReaderDataSetIterator(recordReaderTest,batchSize,labelIndex,numOutputs);
        Evaluation evaluation=new Evaluation();
        while(dataSetIteratorTest.hasNext()){
            DataSet dataSetTest = dataSetIteratorTest.next();
            INDArray features = dataSetTest.getFeatures();
            INDArray targetLabels = dataSetTest.getLabels();
            INDArray predictedLabels = model.output(features);
            evaluation.eval(predictedLabels,targetLabels);


        }
        System.out.println(evaluation.stats());
        ModelSerializer.writeModel(model,"irisModel.zip",true);




    }
}
