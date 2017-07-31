package dk.itu.epbprtv0.algorithm;

public interface QueryAlgorithmFactory {
  boolean configure(String var, String val);
  QueryAlgorithm endConfigure();
}
