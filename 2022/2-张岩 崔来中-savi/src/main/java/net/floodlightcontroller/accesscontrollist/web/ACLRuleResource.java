package net.floodlightcontroller.accesscontrollist.web;
import java.io.IOException;
import java.util.Iterator;
import net.floodlightcontroller.accesscontrollist.ACLRule;
import net.floodlightcontroller.accesscontrollist.IACLService;
import net.floodlightcontroller.accesscontrollist.util.IPAddressUtil;
import org.restlet.resource.Delete;
import org.restlet.resource.Get;
import org.restlet.resource.Post;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.MappingJsonFactory;
public class ACLRuleResource extends ServerResource {
	protected static Logger log = LoggerFactory
			.getLogger(ACLRuleResource.class);
	@Get("json")
	public Object handleRequest() {
		IACLService acl = (IACLService) getContext().getAttributes().get(
				IACLService.class.getCanonicalName());
		return acl.getRules();
	}
	@Post
	public String store(String json) {
		IACLService aclService = (IACLService) getContext().getAttributes().get(
				IACLService.class.getCanonicalName());
		ACLRule newRule;
		try {
			newRule = jsonToRule(json);
		} catch (Exception e) {
			log.error("Error parsing ACL rule: " + json, e);
			return "{\"status\" : \"Failed! " + e.getMessage() + "\"}";
		}
		String status = null;
		String nw_src = newRule.getNw_src();
		String nw_dst = newRule.getNw_dst();
		if (nw_src == null && nw_dst == null){
			status = "Failed! Either nw_src or nw_dst must be specified.";
			return ("{\"status\" : \"" + status + "\"}");
		}
		if(aclService.addRule(newRule)){
			status = "Success! New rule added.";
		}else{
			status = "Failed! The new ACL rule matches an existing rule.";
		}
		return ("{\"status\" : \"" + status + "\"}");
	}
	@Delete
	public String remove(String json) {
		IACLService ACL = (IACLService) getContext().getAttributes().get(
				IACLService.class.getCanonicalName());
		ACLRule rule;
		try {
			rule = jsonToRule(json);
		} catch (Exception e) {
			log.error("Error parsing ACL rule: " + json, e);
			return "{\"status\" : \"Failed! " + e.getMessage() + "\"}";
		}
		boolean exists = false;
		Iterator<ACLRule> iter = ACL.getRules().iterator();
		while (iter.hasNext()) {
			ACLRule r = iter.next();
			if (r.getId() == rule.getId()) {
				exists = true;
				break;
			}
		}
		String status = null;
		if (!exists) {
			status = "Failed! a rule with this ID doesn't exist.";
			log.error(status);
		} else {
			ACL.removeRule(rule.getId());
			status = "Success! Rule deleted";
		}
		return ("{\"status\" : \"" + status + "\"}");
	}
	public static ACLRule jsonToRule(String json) throws IOException {
		ACLRule rule = new ACLRule();
		MappingJsonFactory f = new MappingJsonFactory();
		JsonParser jp;
		try {
			jp = f.createParser(json);
		} catch (JsonParseException e) {
			throw new IOException(e);
		}
		jp.nextToken();
		if (jp.getCurrentToken() != JsonToken.START_OBJECT) {
			throw new IOException("Expected START_OBJECT");
		}
		while (jp.nextToken() != JsonToken.END_OBJECT) {
			if (jp.getCurrentToken() != JsonToken.FIELD_NAME) {
				throw new IOException("Expected FIELD_NAME");
			}
			String key = jp.getCurrentName();
			jp.nextToken();
			String value = jp.getText();
			if (value.equals(""))
				continue;
			if ("ruleid".equals(key)) {
				try{
					rule.setId(Integer.parseInt(value));
				} catch(NumberFormatException e){
					throw new NumberFormatException("ruleid must be specified as a number.");
				}
			}
			else if ("src-ip".equals(key)) {
				rule.setNw_src(value);
				int[] cidr = IPAddressUtil.parseCIDR(value);
				rule.setNw_src_prefix(cidr[0]);
				rule.setNw_src_maskbits(cidr[1]);
			}
			else if ("dst-ip".equals(key)) {
				rule.setNw_dst(value);
				int[] cidr = IPAddressUtil.parseCIDR(value);
				rule.setNw_dst_prefix(cidr[0]);
				rule.setNw_dst_maskbits(cidr[1]);
			}
			else if ("nw-proto".equals(key)) {
				if ("TCP".equalsIgnoreCase(value)) {
					rule.setNw_proto(6);
				} else if ("UDP".equalsIgnoreCase(value)) {
					rule.setNw_proto(11);
				} else if ("ICMP".equalsIgnoreCase(value)) {
					rule.setNw_proto(1);
				} else {
					throw new IllegalArgumentException("nw-proto must be specified as (TCP || UDP || ICMP).");
				}
			}
			else if ("tp-dst".equals(key)) {
				if(rule.getNw_proto() == 6 || rule.getNw_proto() == 11){
					try{
						rule.setTp_dst(Integer.parseInt(value));
					}catch(NumberFormatException e){
						throw new NumberFormatException("tp-dst must be specified as a number.");
					}
				}
			}
			else if (key == "action") {
				if ("allow".equalsIgnoreCase(value)) {
					rule.setAction(ACLRule.Action.ALLOW);
				} else if ("deny".equalsIgnoreCase(value)) {
					rule.setAction(ACLRule.Action.DENY);
				} else{
					throw new IllegalArgumentException("action must be specidied as (allow || deny).");
				}
			}
		}
		return rule;
	}
}
