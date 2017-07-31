package dk.itu.epbprtv0.distance;

import java.util.List;

public class HammingDistance implements DistanceMetric<List<Boolean>> {
  public static final HammingDistance INSTANCE = new HammingDistance();

  @Override
  public double getDistance(List<Boolean> a, List<Boolean> b) {
    assert a.size() == b.size();
    int distance = 0;
    for (int i = 0; i < a.size(); i++) {
      if (a.get(i) != b.get(i))
        distance++;
    }
    return distance;
  }
}
