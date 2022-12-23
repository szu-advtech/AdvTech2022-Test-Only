package net.floodlightcontroller.accesscontrollist.util;
import net.floodlightcontroller.packet.IPv4;
public class IPAddressUtil {
	public static int[] parseCIDR(String cidr) {
		int ret[] = new int[2];
		String[] parts = cidr.split("/");
		if (parts.length == 1){
			throw new IllegalArgumentException("CIDR mask bits must be specified.");
		}
		String cidrPrefix = parts[0].trim();
		int cidrMaskBits = 0;
		if (parts.length == 2) {
			try {
				cidrMaskBits = Integer.parseInt(parts[1].trim());
			} catch (Exception e) {
				throw new NumberFormatException("CIDR mask bits must be specified as a number(0 ~ 32).");
			}
			if (cidrMaskBits < 0 || cidrMaskBits > 32) {
				throw new NumberFormatException("CIDR mask bits must be 0 <= value <= 32.");
			}
		}
		ret[0] = IPv4.toIPv4Address(cidrPrefix);
		ret[1] = cidrMaskBits;
		return ret;
	}
	public static boolean containIP(int cidrPrefix, int cidrMaskBits, int ip) {
		boolean matched = true;
		int bitsToShift = 32 - cidrMaskBits;
		if (bitsToShift > 0) {
			cidrPrefix = cidrPrefix >> bitsToShift;
			ip = ip >> bitsToShift;
			cidrPrefix = cidrPrefix << bitsToShift;
			ip = ip << bitsToShift;
		}
		if (cidrPrefix != ip) {
			matched = false;
		}
		return matched;
	}
	public static boolean isSubnet(String cidr1, String cidr2) {
		if (cidr2 == null) {
			return true;
		} else if (cidr1 == null) {
			return false;
		}
		int[] cidr = IPAddressUtil.parseCIDR(cidr1);
		int cidr1Prefix = cidr[0];
		int cidr1MaskBits = cidr[1];
		cidr = IPAddressUtil.parseCIDR(cidr2);
		int cidr2Prefix = cidr[0];
		int cidr2MaskBits = cidr[1];
		int bitsToShift_1 = 32 - cidr1MaskBits;
		int bitsToShift_2 = 32 - cidr2MaskBits;
		int offset = (bitsToShift_1 > bitsToShift_2) ? bitsToShift_1
				: bitsToShift_2;
		if (offset > 0) {
			cidr1Prefix = cidr1Prefix >> offset;
			cidr2Prefix = cidr2Prefix >> offset;
			cidr1Prefix = cidr1Prefix << offset;
			cidr2Prefix = cidr2Prefix << offset;
		}
		if (cidr1Prefix == cidr2Prefix) {
			if (cidr1MaskBits >= cidr2MaskBits) {
				return true;
			}
		}
		return false;
	}
}
