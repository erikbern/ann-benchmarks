package dk.itu.epbprtv0.types;

import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

public class BitInfo implements TypeInfo<List<Boolean>> {
  public static final BitInfo INSTANCE = new BitInfo();

  @Override
  public List<Boolean> parse(String item) {
    ArrayList<Boolean> results = new ArrayList<Boolean>();
    for (int i = 0; i < item.length(); i++) {
      char c = item.charAt(i);
      if (c == '0') {
        results.add(false);
      } else if (c == '1') {
        results.add(true);
      }
    }
    return results;
  }

  @Override
  public String unparse(List<Boolean> item) {
    StringBuilder sb = new StringBuilder();
    Iterator<Boolean> it = item.iterator();
    while (it.hasNext()) {
      sb.append(it.next() ? '1' : '0');
      if (it.hasNext())
        sb.append(' ');
    }
    return sb.toString();
  }

  public static void main(String[] args) {
    List<Boolean> h = BitInfo.INSTANCE.parse("1101100100100100");
    System.out.println(BitInfo.INSTANCE.unparse(h));
  }
}