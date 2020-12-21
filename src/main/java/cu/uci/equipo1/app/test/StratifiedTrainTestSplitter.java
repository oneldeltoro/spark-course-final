package cu.uci.equipo1.app.test;

import cu.uci.equipo1.app.AlgorithmsBase;
import lombok.NoArgsConstructor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Optional;

/**
 * <p> "Escriba su texto aquí"</p>
 * Author: Onel Del Toro Rodríguez <a href="mailto>:onel.deltoro@datys.cu">onel.deltoro@datys.cu</a>
 */
@NoArgsConstructor
public class StratifiedTrainTestSplitter {

    public Dataset<Row>[] randomSplit(SparkSession spark, Dataset<Row> datasetBruto, String[] labels, double[] weights) {
        Dataset<Row>[] result = null;

        for (String label : labels) {
            if (result == null) {
                result = getPorcent(spark, datasetBruto, label, null);
            } else {
                Dataset<Row>[] tmp = getPorcent(spark, datasetBruto, label, null);
                Dataset<Row> r0 = result[0].union(tmp[0]);
                Dataset<Row> r1 = result[1].union(tmp[1]);
                result = new Dataset[]{r0, r1};
            }
        }
        System.out.println("traing"+result[0].count());
        System.out.println("test"+result[1].count());
        return result;
    }

    public Dataset<Row>[] getPorcent(SparkSession spark, Dataset<Row> datasetBruto, String labels, double[] weights) {

        datasetBruto.createOrReplaceTempView("labels");
        String query = "Select * from labels where class='" + labels + "'";

        Dataset<Row> inciso4 = spark.sql(query);
        System.out.println("Cantidad de: " + labels + " son: " + inciso4.count());
        return AlgorithmsBase.getDatasets(inciso4, Optional.of(new double[]{0.7, 0.3}), Optional.of(12345L));
    }
}
