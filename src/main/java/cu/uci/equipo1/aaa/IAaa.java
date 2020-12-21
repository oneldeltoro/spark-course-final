/*
 * Interface para algoritmos de aprendisaje automaticos
 */
package cu.uci.equipo1.aaa;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.tuning.TrainValidationSplitParams;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 *
 * @author Liban
 */
/**
 * *
 *
 * @author Liban
 */
public interface IAaa {

    /**
     * Transforma y extrae caracteristicas
     *
     * 
     * @param df sobre el que se ejecutan ala transformaciones
     * @return df transformado
     */
    Pipeline extractFeacture(Dataset df);

    /***
     * Llama al algorimos de clasifiación o predición sobre TDD
     * @param df
     * @return 
     */
    TrainValidationSplitParams practicing(Dataset df);

    /***
     * Utiliza la funcion de evaluacion de mlib para evaluar el modelo 
     * en el conjunto de datos de prueba
     */
    Dataset<Row> evaluate();

}
