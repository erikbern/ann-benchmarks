package dk.itu.epbprtv0.algorithm;

import java.util.List;

public interface QueryAlgorithm {
  boolean train(String entry);
  void endTrain();

  List<Integer> query(String entry, int count);
}
