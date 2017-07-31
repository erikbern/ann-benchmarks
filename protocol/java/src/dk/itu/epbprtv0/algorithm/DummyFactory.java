package dk.itu.epbprtv0.algorithm;

import java.util.List;
import java.util.Collections;

public class DummyFactory implements QueryAlgorithmFactory {
  private static final class DummyAlgorithm implements QueryAlgorithm {
    @Override
    public boolean train(String item) {
      return true;
    }

    @Override
    public void endTrain() {
    }

    @Override
    public List<Integer> query(String item, int k) {
      return Collections.emptyList();
    }

    public static final DummyAlgorithm INSTANCE = new DummyAlgorithm();
  }

  @Override
  public boolean configure(String var, String val) {
    return true;
  }

  @Override
  public QueryAlgorithm endConfigure() {
    return DummyAlgorithm.INSTANCE;
  }
}