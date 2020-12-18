package cu.uci.equipo1.app.test;

import lombok.NoArgsConstructor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 * <p> "Escriba su texto aquí"</p>
 * Author: Onel Del Toro Rodríguez <a href="mailto>:onel.deltoro@datys.cu">onel.deltoro@datys.cu</a>
 */
@NoArgsConstructor
public class StratifiedTrainTestSplitter {

    static Dataset<Row>[] randomSplit(Dataset<Row> datasetBruto, Iterable<String> labels, double[] weights) {

        return new Dataset[0];
    }


}
